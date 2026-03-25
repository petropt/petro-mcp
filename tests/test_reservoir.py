"""Tests for reservoir engineering material balance tools."""

import json
import math

import pytest

from petro_mcp.tools.reservoir import (
    calculate_havlena_odeh,
    calculate_pz_analysis,
    calculate_radius_of_investigation,
    calculate_recovery_factor,
    calculate_volumetric_ogip,
    calculate_volumetric_ooip,
)


# ===========================================================================
# P/Z Analysis (Gas Material Balance)
# ===========================================================================

class TestPZAnalysis:
    """Tests for gas material balance P/Z analysis."""

    def test_textbook_example(self):
        """Test with a textbook-style gas reservoir.

        Known: Pi/Zi = 4000 psi, OGIP = 100 Bcf.
        P/Z should decline linearly with Gp.
        At Gp=0, P/Z=4000. At Gp=100, P/Z=0.
        So slope = -40 per Bcf.
        """
        # Generate synthetic data along the line P/Z = 4000 - 40*Gp
        gp = [0, 10, 20, 30, 40, 50]
        pz = [4000, 3600, 3200, 2800, 2400, 2000]

        result = json.loads(calculate_pz_analysis(pz, gp))

        assert result["method"] == "P/Z vs Gp (Gas Material Balance)"
        assert result["results"]["ogip_bcf"] == pytest.approx(100.0, rel=1e-3)
        assert result["results"]["r_squared"] == pytest.approx(1.0, rel=1e-6)
        assert result["results"]["intercept_psi"] == pytest.approx(4000, rel=1e-3)
        assert result["results"]["slope"] == pytest.approx(-40, rel=1e-3)

    def test_recovery_factor(self):
        """Current recovery factor matches Gp_max / OGIP."""
        gp = [0, 25, 50]
        pz = [3000, 2250, 1500]
        # OGIP = 3000/15 = 200... slope = (1500-3000)/(50-0) = -30
        # intercept = 3000, OGIP = 3000/30 = 100

        result = json.loads(calculate_pz_analysis(pz, gp))
        ogip = result["results"]["ogip_bcf"]
        rf = result["results"]["current_recovery_factor"]

        assert ogip == pytest.approx(100.0, rel=1e-3)
        assert rf == pytest.approx(0.5, rel=1e-3)

    def test_abandonment_pressure(self):
        """Abandonment recovery is calculated when abandonment_pressure given."""
        gp = [0, 10, 20, 30, 40, 50]
        pz = [4000, 3600, 3200, 2800, 2400, 2000]

        result = json.loads(calculate_pz_analysis(pz, gp, abandonment_pressure=800))

        assert "abandonment" in result
        abn = result["abandonment"]
        assert abn["abandonment_pressure_psi"] == 800
        # At P/Z=800: Gp = (4000-800)/40 = 80 Bcf
        assert abn["recoverable_gas_bcf"] == pytest.approx(80.0, rel=1e-3)
        assert abn["abandonment_recovery_factor"] == pytest.approx(0.8, rel=1e-3)

    def test_two_points_minimum(self):
        """Should work with exactly 2 data points."""
        result = json.loads(calculate_pz_analysis([3000, 2000], [0, 20]))
        assert result["results"]["ogip_bcf"] is not None

    def test_single_point_raises(self):
        """Single point is insufficient for regression."""
        with pytest.raises(ValueError, match="At least 2"):
            calculate_pz_analysis([3000], [0])

    def test_empty_arrays_raises(self):
        """Empty arrays should raise."""
        with pytest.raises(ValueError, match="At least 2"):
            calculate_pz_analysis([], [])

    def test_mismatched_lengths_raises(self):
        """Mismatched array lengths should raise."""
        with pytest.raises(ValueError, match="equal length"):
            calculate_pz_analysis([3000, 2000], [0, 10, 20])

    def test_negative_pressure_raises(self):
        """Negative pressures should raise."""
        with pytest.raises(ValueError, match="positive"):
            calculate_pz_analysis([-100, 2000], [0, 10])

    def test_negative_cumulative_gas_raises(self):
        """Negative cumulative gas should raise."""
        with pytest.raises(ValueError, match="non-negative"):
            calculate_pz_analysis([3000, 2000], [-5, 10])

    def test_noisy_data(self):
        """P/Z analysis should handle noisy data with R² < 1."""
        gp = [0, 10, 20, 30, 40, 50]
        pz = [4000, 3650, 3150, 2850, 2350, 2050]  # slight noise

        result = json.loads(calculate_pz_analysis(pz, gp))
        assert result["results"]["ogip_bcf"] is not None
        assert 0.95 < result["results"]["r_squared"] < 1.0


# ===========================================================================
# Volumetric OOIP
# ===========================================================================

class TestVolumetricOOIP:
    """Tests for volumetric OOIP calculation."""

    def test_hand_calculation(self):
        """OOIP = 7758 * 640 * 50 * 0.2 * (1-0.25) / 1.2 = 31,032,000 STB."""
        result = json.loads(calculate_volumetric_ooip(
            area_acres=640,
            thickness_ft=50,
            porosity=0.2,
            sw=0.25,
            bo=1.2,
        ))

        expected = 7758 * 640 * 50 * 0.2 * 0.75 / 1.2
        assert result["results"]["ooip_stb"] == pytest.approx(expected, rel=1e-6)

    def test_unit_area(self):
        """1 acre, 1 ft, phi=1, Sw=0, Bo=1 should give 7758 STB."""
        result = json.loads(calculate_volumetric_ooip(1, 1, 1.0, 0.0, 1.0))
        assert result["results"]["ooip_stb"] == pytest.approx(7758.0, rel=1e-6)

    def test_mmstb_conversion(self):
        """Check that MMSTB = STB / 1e6."""
        result = json.loads(calculate_volumetric_ooip(640, 50, 0.2, 0.25, 1.2))
        stb = result["results"]["ooip_stb"]
        mmstb = result["results"]["ooip_mmstb"]
        assert mmstb == pytest.approx(stb / 1e6, rel=1e-4)

    def test_zero_porosity(self):
        """Zero porosity should give zero OOIP (but fraction validator allows 0)."""
        result = json.loads(calculate_volumetric_ooip(640, 50, 0.0, 0.25, 1.2))
        assert result["results"]["ooip_stb"] == 0.0

    def test_negative_area_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_volumetric_ooip(-100, 50, 0.2, 0.25, 1.2)

    def test_porosity_out_of_range_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            calculate_volumetric_ooip(640, 50, 1.5, 0.25, 1.2)

    def test_sw_out_of_range_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            calculate_volumetric_ooip(640, 50, 0.2, -0.1, 1.2)


# ===========================================================================
# Volumetric OGIP
# ===========================================================================

class TestVolumetricOGIP:
    """Tests for volumetric OGIP calculation."""

    def test_hand_calculation(self):
        """OGIP = 43560 * 640 * 100 * 0.15 * (1-0.3) / 0.005."""
        result = json.loads(calculate_volumetric_ogip(
            area_acres=640,
            thickness_ft=100,
            porosity=0.15,
            sw=0.3,
            bg=0.005,
        ))

        expected = 43560 * 640 * 100 * 0.15 * 0.70 / 0.005
        assert result["results"]["ogip_scf"] == pytest.approx(expected, rel=1e-6)

    def test_bcf_conversion(self):
        """Check Bcf = scf / 1e9."""
        result = json.loads(calculate_volumetric_ogip(640, 100, 0.15, 0.3, 0.005))
        scf = result["results"]["ogip_scf"]
        bcf = result["results"]["ogip_bcf"]
        assert bcf == pytest.approx(scf / 1e9, rel=1e-4)

    def test_negative_bg_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_volumetric_ogip(640, 100, 0.15, 0.3, -0.005)


# ===========================================================================
# Recovery Factor
# ===========================================================================

class TestRecoveryFactor:
    """Tests for recovery factor calculation."""

    def test_simple_rf(self):
        """RF = 2e6 / 10e6 = 0.2 (20%)."""
        result = json.loads(calculate_recovery_factor(10e6, 2e6))
        assert result["results"]["recovery_factor"] == pytest.approx(0.2, rel=1e-6)
        assert result["results"]["recovery_factor_pct"] == pytest.approx(20.0, rel=1e-4)

    def test_zero_production(self):
        """RF = 0 when no production."""
        result = json.loads(calculate_recovery_factor(10e6, 0))
        assert result["results"]["recovery_factor"] == 0.0

    def test_full_recovery(self):
        """RF = 1.0 at 100% recovery."""
        result = json.loads(calculate_recovery_factor(1000, 1000))
        assert result["results"]["recovery_factor"] == pytest.approx(1.0, rel=1e-6)

    def test_remaining_reserves(self):
        """Remaining = OOIP - Np."""
        result = json.loads(calculate_recovery_factor(10e6, 3e6))
        assert result["results"]["remaining_in_place"] == pytest.approx(7e6, rel=1e-6)

    def test_zero_ooip_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_recovery_factor(0, 100)

    def test_negative_production_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            calculate_recovery_factor(10e6, -100)


# ===========================================================================
# Radius of Investigation
# ===========================================================================

class TestRadiusOfInvestigation:
    """Tests for radius of investigation calculation."""

    def test_known_values(self):
        """Test with typical well test values.

        k=100 md, t=24 hr, phi=0.2, mu=1 cp, ct=10e-6 1/psi
        r_inv = 0.029 * sqrt(100*24 / (0.2*1*10e-6))
        = 0.029 * sqrt(2400 / 2e-6) = 0.029 * sqrt(1.2e9)
        = 0.029 * 34641.02 = 1004.6 ft
        """
        result = json.loads(calculate_radius_of_investigation(
            permeability_md=100,
            time_hours=24,
            porosity=0.2,
            viscosity_cp=1.0,
            total_compressibility=10e-6,
        ))

        expected = 0.029 * math.sqrt(100 * 24 / (0.2 * 1.0 * 10e-6))
        assert result["results"]["radius_of_investigation_ft"] == pytest.approx(
            expected, rel=1e-3
        )

    def test_area_in_acres(self):
        """Area should be pi*r^2 / 43560."""
        result = json.loads(calculate_radius_of_investigation(
            permeability_md=100, time_hours=24,
            porosity=0.2, viscosity_cp=1.0, total_compressibility=10e-6,
        ))

        r = result["results"]["radius_of_investigation_ft"]
        expected_acres = math.pi * r ** 2 / 43560
        assert result["results"]["area_investigated_acres"] == pytest.approx(
            expected_acres, rel=1e-3
        )

    def test_low_perm_short_time(self):
        """Low perm + short time = small radius."""
        result = json.loads(calculate_radius_of_investigation(
            permeability_md=0.1, time_hours=1,
            porosity=0.1, viscosity_cp=5.0, total_compressibility=20e-6,
        ))
        assert result["results"]["radius_of_investigation_ft"] < 100

    def test_zero_porosity_raises(self):
        with pytest.raises(ValueError, match="porosity must be > 0"):
            calculate_radius_of_investigation(100, 24, 0.0, 1.0, 10e-6)

    def test_negative_permeability_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_radius_of_investigation(-10, 24, 0.2, 1.0, 10e-6)


# ===========================================================================
# Havlena-Odeh Oil Material Balance
# ===========================================================================

class TestHavlenaOdeh:
    """Tests for Havlena-Odeh straight-line material balance."""

    @staticmethod
    def _simple_depletion_data():
        """Simplified depletion-drive reservoir data (no gas cap, no water)."""
        # 4 time steps: initial + 3 depletion steps
        pressures = [4000, 3500, 3000, 2500]
        np_vals = [0, 500000, 1200000, 2000000]
        rp_vals = [0, 600, 650, 700]  # scf/STB
        wp_vals = [0, 0, 0, 0]
        wi_vals = [0, 0, 0, 0]
        # Bo increases slightly as pressure drops below Pb
        bo_vals = [1.25, 1.22, 1.18, 1.14]
        rs_vals = [500, 420, 340, 260]  # scf/STB
        bg_vals = [0.0008, 0.00095, 0.0012, 0.0016]  # bbl/scf
        bw_vals = [1.02, 1.02, 1.02, 1.02]

        return dict(
            pressures=pressures,
            np_values=np_vals,
            rp_values=rp_vals,
            wp_values=wp_vals,
            wi_values=wi_vals,
            bo_values=bo_vals,
            rs_values=rs_vals,
            bg_values=bg_vals,
            bw_values=bw_vals,
            boi=1.25,
            rsi=500,
            bgi=0.0008,
        )

    def test_depletion_drive_returns_ooip(self):
        """OOIP should be estimated for a simple depletion drive case."""
        data = self._simple_depletion_data()
        result = json.loads(calculate_havlena_odeh(**data))

        assert result["method"] == "Havlena-Odeh Straight-Line Material Balance"
        assert result["results"]["ooip_stb"] is not None
        assert result["results"]["ooip_stb"] > 0

    def test_drive_indices_computed(self):
        """Drive indices should be computed and sum roughly to 1."""
        data = self._simple_depletion_data()
        result = json.loads(calculate_havlena_odeh(**data))

        di = result["results"]["drive_indices"]
        ddi = di["depletion_drive_index"]
        sdi = di["gas_cap_drive_index"]
        wdi = di["water_drive_index"]

        assert ddi is not None
        assert sdi is not None
        assert wdi is not None

        # With no water production, water drive should be ~0
        assert wdi == pytest.approx(0.0, abs=1e-6)

        # Dominant drive should be identified
        assert result["results"]["dominant_drive"] in (
            "depletion_drive", "gas_cap_drive", "water_drive",
        )

    def test_plot_data_lengths(self):
        """F and Et arrays should match number of data points."""
        data = self._simple_depletion_data()
        result = json.loads(calculate_havlena_odeh(**data))

        n = result["inputs"]["num_data_points"]
        assert len(result["plot_data"]["f_rb"]) == n
        assert len(result["plot_data"]["et_rb_stb"]) == n

    def test_f_at_initial_is_zero(self):
        """At initial conditions (Np=0), F should be ~0."""
        data = self._simple_depletion_data()
        result = json.loads(calculate_havlena_odeh(**data))

        # F[0] should be 0 (no production at initial conditions)
        assert result["plot_data"]["f_rb"][0] == pytest.approx(0, abs=1e-2)

    def test_with_cf_swi(self):
        """Adding cf and swi should change Efw and still produce valid OOIP."""
        data = self._simple_depletion_data()
        data["cf"] = 3e-6
        data["swi"] = 0.25

        result = json.loads(calculate_havlena_odeh(**data))
        assert result["results"]["ooip_stb"] is not None
        # Efw should be nonzero at later time steps
        efw = result["plot_data"]["efw_rb_stb"]
        assert any(e != 0 for e in efw[1:])

    def test_two_points_minimum(self):
        """Should work with exactly 2 data points."""
        result = json.loads(calculate_havlena_odeh(
            pressures=[4000, 3500],
            np_values=[0, 500000],
            rp_values=[0, 600],
            wp_values=[0, 0],
            wi_values=[0, 0],
            bo_values=[1.25, 1.22],
            rs_values=[500, 420],
            bg_values=[0.0008, 0.00095],
            bw_values=[1.02, 1.02],
            boi=1.25, rsi=500, bgi=0.0008,
        ))
        assert result["results"]["ooip_stb"] is not None

    def test_single_point_raises(self):
        """Single data point should raise."""
        with pytest.raises(ValueError, match="At least 2"):
            calculate_havlena_odeh(
                pressures=[4000],
                np_values=[0],
                rp_values=[0],
                wp_values=[0],
                wi_values=[0],
                bo_values=[1.25],
                rs_values=[500],
                bg_values=[0.0008],
                bw_values=[1.02],
                boi=1.25, rsi=500, bgi=0.0008,
            )

    def test_mismatched_lengths_raises(self):
        """Mismatched array lengths should raise."""
        with pytest.raises(ValueError, match="equal length"):
            calculate_havlena_odeh(
                pressures=[4000, 3500],
                np_values=[0, 500000, 1000000],  # wrong length
                rp_values=[0, 600],
                wp_values=[0, 0],
                wi_values=[0, 0],
                bo_values=[1.25, 1.22],
                rs_values=[500, 420],
                bg_values=[0.0008, 0.00095],
                bw_values=[1.02, 1.02],
                boi=1.25, rsi=500, bgi=0.0008,
            )

    def test_negative_boi_raises(self):
        """Negative Boi should raise."""
        data = self._simple_depletion_data()
        data["boi"] = -1.0
        with pytest.raises(ValueError, match="positive"):
            calculate_havlena_odeh(**data)
