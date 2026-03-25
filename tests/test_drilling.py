"""Tests for drilling engineering calculations."""

import json
import math

import pytest

from petro_mcp.tools.drilling import (
    calculate_annular_velocity,
    calculate_bit_pressure_drop,
    calculate_burst_pressure,
    calculate_collapse_pressure,
    calculate_ecd,
    calculate_formation_pressure_gradient,
    calculate_hydrostatic_pressure,
    calculate_icp_fcp,
    calculate_kill_mud_weight,
    calculate_maasp,
    calculate_nozzle_tfa,
)


# -----------------------------------------------------------------------
# Hydrostatic pressure
# -----------------------------------------------------------------------

class TestHydrostaticPressure:
    def test_basic(self):
        # 10 ppg at 10000 ft: 0.052 * 10 * 10000 = 5200 psi
        result = json.loads(calculate_hydrostatic_pressure(10.0, 10000.0))
        assert result["hydrostatic_pressure_psi"] == 5200.0
        assert "formula" in result

    def test_fresh_water(self):
        # Fresh water ~8.33 ppg at 1000 ft: 0.052 * 8.33 * 1000 = 433.16
        result = json.loads(calculate_hydrostatic_pressure(8.33, 1000.0))
        assert abs(result["hydrostatic_pressure_psi"] - 433.16) < 0.01

    def test_heavy_mud(self):
        # 16 ppg at 15000 ft: 0.052 * 16 * 15000 = 12480
        result = json.loads(calculate_hydrostatic_pressure(16.0, 15000.0))
        assert result["hydrostatic_pressure_psi"] == 12480.0

    def test_zero_mud_weight_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            calculate_hydrostatic_pressure(0.0, 10000.0)

    def test_negative_tvd_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            calculate_hydrostatic_pressure(10.0, -100.0)


# -----------------------------------------------------------------------
# ECD
# -----------------------------------------------------------------------

class TestECD:
    def test_basic(self):
        # MW=10, APL=200, TVD=10000: ECD = 10 + 200/(0.052*10000) = 10 + 0.3846 = 10.3846
        result = json.loads(calculate_ecd(10.0, 200.0, 10000.0))
        assert abs(result["ecd_ppg"] - 10.3846) < 0.001

    def test_zero_apl(self):
        # No annular losses: ECD = MW
        result = json.loads(calculate_ecd(12.0, 0.0, 8000.0))
        assert result["ecd_ppg"] == 12.0

    def test_negative_apl_raises(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            calculate_ecd(10.0, -50.0, 10000.0)


# -----------------------------------------------------------------------
# Formation pressure gradient
# -----------------------------------------------------------------------

class TestFormationPressureGradient:
    def test_normal_gradient(self):
        # Normal pore pressure: ~4333 psi at 10000 ft -> 4333/(0.052*10000) = 8.333 ppg
        result = json.loads(calculate_formation_pressure_gradient(4333.0, 10000.0))
        assert abs(result["formation_pressure_gradient_ppg"] - 8.3327) < 0.01

    def test_overpressured(self):
        # 6500 psi at 10000 ft -> 6500/520 = 12.5 ppg
        result = json.loads(calculate_formation_pressure_gradient(6500.0, 10000.0))
        assert abs(result["formation_pressure_gradient_ppg"] - 12.5) < 0.001

    def test_zero_pressure(self):
        result = json.loads(calculate_formation_pressure_gradient(0.0, 10000.0))
        assert result["formation_pressure_gradient_ppg"] == 0.0


# -----------------------------------------------------------------------
# Kill mud weight
# -----------------------------------------------------------------------

class TestKillMudWeight:
    def test_basic(self):
        # SIDP=500, MW=10, TVD=10000: Kill MW = 10 + 500/520 = 10.9615
        result = json.loads(calculate_kill_mud_weight(500.0, 10.0, 10000.0))
        assert abs(result["kill_mud_weight_ppg"] - 10.9615) < 0.001

    def test_zero_sidp(self):
        # No kick: kill MW = original MW
        result = json.loads(calculate_kill_mud_weight(0.0, 12.0, 8000.0))
        assert result["kill_mud_weight_ppg"] == 12.0

    def test_high_sidp(self):
        # SIDP=1000, MW=9.5, TVD=12000: 9.5 + 1000/(0.052*12000) = 9.5 + 1.6026 = 11.1026
        result = json.loads(calculate_kill_mud_weight(1000.0, 9.5, 12000.0))
        assert abs(result["kill_mud_weight_ppg"] - 11.1026) < 0.001


# -----------------------------------------------------------------------
# ICP and FCP
# -----------------------------------------------------------------------

class TestICPFCP:
    def test_basic(self):
        # SIDP=500, SCP=400, Kill MW=11.0, Original MW=10.0
        # ICP = 500 + 400 = 900
        # FCP = 400 * (11/10) = 440
        result = json.loads(calculate_icp_fcp(500.0, 400.0, 11.0, 10.0))
        assert result["icp_psi"] == 900.0
        assert result["fcp_psi"] == 440.0

    def test_same_mud_weight(self):
        # If kill MW = original MW, FCP = SCP
        result = json.loads(calculate_icp_fcp(200.0, 300.0, 10.0, 10.0))
        assert result["icp_psi"] == 500.0
        assert result["fcp_psi"] == 300.0

    def test_zero_sidp(self):
        result = json.loads(calculate_icp_fcp(0.0, 350.0, 11.5, 10.0))
        assert result["icp_psi"] == 350.0
        assert abs(result["fcp_psi"] - 350.0 * 11.5 / 10.0) < 0.01


# -----------------------------------------------------------------------
# MAASP
# -----------------------------------------------------------------------

class TestMAASP:
    def test_basic(self):
        # FG=14.5, MW=10.0, shoe TVD=5000
        # MAASP = (14.5 - 10.0) * 0.052 * 5000 = 4.5 * 260 = 1170
        result = json.loads(calculate_maasp(14.5, 10.0, 5000.0))
        assert result["maasp_psi"] == 1170.0

    def test_narrow_margin(self):
        # FG=10.5, MW=10.0, shoe TVD=3000
        # MAASP = 0.5 * 0.052 * 3000 = 78
        result = json.loads(calculate_maasp(10.5, 10.0, 3000.0))
        assert result["maasp_psi"] == 78.0

    def test_negative_margin(self):
        # MW > FG is possible (overbalanced beyond fracture)
        # Result should be negative
        result = json.loads(calculate_maasp(10.0, 12.0, 5000.0))
        assert result["maasp_psi"] < 0


# -----------------------------------------------------------------------
# Annular velocity
# -----------------------------------------------------------------------

class TestAnnularVelocity:
    def test_basic(self):
        # Q=400 gpm, Dh=8.5 in, Dp=5.0 in
        # AV = 24.51 * 400 / (8.5^2 - 5^2) = 9804 / (72.25 - 25) = 9804 / 47.25 = 207.49
        result = json.loads(calculate_annular_velocity(400.0, 8.5, 5.0))
        assert abs(result["annular_velocity_ft_per_min"] - 207.49) < 0.1

    def test_large_annulus(self):
        # Q=600, Dh=12.25, Dp=5.0
        # AV = 24.51 * 600 / (150.0625 - 25) = 14706 / 125.0625 = 117.59
        result = json.loads(calculate_annular_velocity(600.0, 12.25, 5.0))
        assert abs(result["annular_velocity_ft_per_min"] - 117.59) < 0.1

    def test_pipe_equals_hole_raises(self):
        with pytest.raises(ValueError, match="must be less than"):
            calculate_annular_velocity(400.0, 5.0, 5.0)

    def test_pipe_larger_than_hole_raises(self):
        with pytest.raises(ValueError, match="must be less than"):
            calculate_annular_velocity(400.0, 5.0, 6.0)


# -----------------------------------------------------------------------
# Nozzle TFA
# -----------------------------------------------------------------------

class TestNozzleTFA:
    def test_three_12s(self):
        # 3 x 12/32" nozzles: each area = pi/4 * (12/32)^2 = pi/4 * 0.140625 = 0.11045
        # TFA = 3 * 0.11045 = 0.33134
        result = json.loads(calculate_nozzle_tfa([12, 12, 12]))
        expected = 3 * math.pi / 4 * (12 / 32.0) ** 2
        assert abs(result["tfa_sqin"] - round(expected, 4)) < 0.001
        assert result["nozzle_count"] == 3

    def test_mixed_nozzles(self):
        # 3 x 11 + 3 x 12
        result = json.loads(calculate_nozzle_tfa([11, 11, 11, 12, 12, 12]))
        expected = 3 * math.pi / 4 * (11 / 32.0) ** 2 + 3 * math.pi / 4 * (12 / 32.0) ** 2
        assert abs(result["tfa_sqin"] - round(expected, 4)) < 0.001
        assert result["nozzle_count"] == 6

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            calculate_nozzle_tfa([])

    def test_zero_nozzle_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            calculate_nozzle_tfa([12, 0, 12])


# -----------------------------------------------------------------------
# Bit pressure drop
# -----------------------------------------------------------------------

class TestBitPressureDrop:
    def test_basic(self):
        # MW=10, Q=400, TFA=0.3314
        # dP = 10 * 400^2 / (12032 * 0.3314^2) = 1600000 / 1321.04 = 1211.14
        result = json.loads(calculate_bit_pressure_drop(400.0, 10.0, 0.3314))
        expected = 10.0 * 400.0 ** 2 / (12032.0 * 0.3314 ** 2)
        assert abs(result["bit_pressure_drop_psi"] - round(expected, 2)) < 1.0

    def test_higher_flow_rate(self):
        # Pressure drop scales with Q^2
        result_low = json.loads(calculate_bit_pressure_drop(200.0, 10.0, 0.3))
        result_high = json.loads(calculate_bit_pressure_drop(400.0, 10.0, 0.3))
        assert abs(result_high["bit_pressure_drop_psi"] / result_low["bit_pressure_drop_psi"] - 4.0) < 0.01

    def test_zero_tfa_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            calculate_bit_pressure_drop(400.0, 10.0, 0.0)


# -----------------------------------------------------------------------
# Burst pressure (Barlow)
# -----------------------------------------------------------------------

class TestBurstPressure:
    def test_basic(self):
        # 9-5/8" casing, 47 lb/ft, N-80 (80000 psi YS), t=0.472"
        # P = 0.875 * 2 * 80000 * 0.472 / 9.625 = 6880.0
        result = json.loads(calculate_burst_pressure(80000.0, 0.472, 9.625))
        expected = 0.875 * 2.0 * 80000.0 * 0.472 / 9.625
        assert abs(result["burst_pressure_psi"] - round(expected, 2)) < 0.1

    def test_small_tubing(self):
        # 2-7/8" tubing, L-80 (80000 psi), t=0.217"
        result = json.loads(calculate_burst_pressure(80000.0, 0.217, 2.875))
        expected = 0.875 * 2.0 * 80000.0 * 0.217 / 2.875
        assert abs(result["burst_pressure_psi"] - round(expected, 2)) < 0.1

    def test_wall_too_thick_raises(self):
        with pytest.raises(ValueError, match="must be less than"):
            calculate_burst_pressure(80000.0, 5.0, 9.625)

    def test_zero_yield_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            calculate_burst_pressure(0.0, 0.472, 9.625)


# -----------------------------------------------------------------------
# Collapse pressure (API 5C3)
# -----------------------------------------------------------------------

class TestCollapsePressure:
    def test_returns_positive(self):
        # 9-5/8" 47 lb/ft N-80: OD=9.625, t=0.472, Fy=80000
        result = json.loads(calculate_collapse_pressure(9.625, 0.472, 80000.0, "N-80"))
        assert result["collapse_pressure_psi"] > 0
        assert result["regime"] in {"yield", "plastic", "transition", "elastic"}
        assert result["method"] == "API 5C3"
        assert result["inputs"]["grade"] == "N-80"

    def test_thick_wall_yield_regime(self):
        # Very thick wall -> low D/t -> yield collapse
        result = json.loads(calculate_collapse_pressure(7.0, 1.0, 80000.0))
        assert result["regime"] == "yield"
        # Yield collapse: P = 2*Fy*((D/t - 1)/(D/t)^2)
        dt = 7.0 / 1.0
        expected = 2.0 * 80000.0 * ((dt - 1) / dt ** 2)
        assert abs(result["collapse_pressure_psi"] - round(expected, 2)) < 1.0

    def test_thin_wall_elastic_regime(self):
        # Very thin wall -> high D/t -> elastic collapse
        result = json.loads(calculate_collapse_pressure(20.0, 0.25, 55000.0))
        dt = 20.0 / 0.25  # D/t = 80
        assert result["regime"] == "elastic"
        expected = 46.95e6 / (dt * (dt - 1) ** 2)
        assert abs(result["collapse_pressure_psi"] - round(expected, 2)) < 1.0

    def test_d_over_t_in_output(self):
        result = json.loads(calculate_collapse_pressure(9.625, 0.472, 80000.0))
        assert abs(result["d_over_t"] - 9.625 / 0.472) < 0.01

    def test_j55_typical(self):
        # J-55 casing: Fy=55000
        result = json.loads(calculate_collapse_pressure(7.0, 0.362, 55000.0, "J-55"))
        assert result["collapse_pressure_psi"] > 0
        assert result["inputs"]["grade"] == "J-55"

    def test_zero_wall_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            calculate_collapse_pressure(9.625, 0.0, 80000.0)

    def test_wall_too_thick_raises(self):
        with pytest.raises(ValueError, match="must be less than"):
            calculate_collapse_pressure(9.625, 5.0, 80000.0)

    def test_p110_typical(self):
        # P-110 casing: Fy=110000
        result = json.loads(calculate_collapse_pressure(9.625, 0.472, 110000.0, "P-110"))
        assert result["collapse_pressure_psi"] > 0

    def test_grade_label_optional(self):
        result = json.loads(calculate_collapse_pressure(9.625, 0.472, 80000.0))
        assert "grade" not in result["inputs"]
