"""Tests for Rate Transient Analysis (RTA) tools."""

import json
import math

import pytest

from petro_mcp.tools.rta import (
    calculate_normalized_rate,
    calculate_material_balance_time,
    calculate_blasingame_variables,
    calculate_agarwal_gardner_variables,
    calculate_npi_variables,
    calculate_flowing_material_balance,
    calculate_sqrt_time_analysis,
    calculate_rta_permeability,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic production data
# ---------------------------------------------------------------------------

def _make_decline_data(n=30, qi=500.0, Di=0.05, pi=3000.0, pwf_base=1500.0):
    """Generate synthetic exponential-decline data with slight pressure drawdown."""
    import numpy as np
    t = list(range(1, n + 1))
    rates = [qi * math.exp(-Di * ti) for ti in t]
    cum = []
    c = 0.0
    for i, r in enumerate(rates):
        c += r  # unit time steps
        cum.append(c)
    # Pressure drops slightly over time
    pwf = [pwf_base - 10 * i for i in range(n)]
    return t, rates, cum, pwf, pi


# ---------------------------------------------------------------------------
# calculate_normalized_rate
# ---------------------------------------------------------------------------

class TestNormalizedRate:
    def test_basic(self):
        result = json.loads(calculate_normalized_rate(
            [100, 80, 60], [2000, 2100, 2200], 3000.0,
        ))
        assert result["num_points"] == 3
        assert len(result["normalized_rate"]) == 3
        # q / (Pi - Pwf): 100 / 1000 = 0.1
        assert abs(result["normalized_rate"][0] - 0.1) < 1e-4

    def test_zero_drawdown_gives_zero(self):
        result = json.loads(calculate_normalized_rate(
            [100, 80], [3000, 2500], 3000.0,
        ))
        # First point: dp = 0 -> normalized = 0
        assert result["normalized_rate"][0] == 0.0
        # Second point: dp = 500 -> 80/500 = 0.16
        assert abs(result["normalized_rate"][1] - 0.16) < 1e-4

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            calculate_normalized_rate([100, 80], [2000], 3000.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="not be empty"):
            calculate_normalized_rate([], [], 3000.0)

    def test_negative_initial_pressure_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_normalized_rate([100], [2000], -100.0)


# ---------------------------------------------------------------------------
# calculate_material_balance_time
# ---------------------------------------------------------------------------

class TestMaterialBalanceTime:
    def test_basic(self):
        result = json.loads(calculate_material_balance_time(
            [100, 250, 420], [100, 80, 60],
        ))
        assert result["num_points"] == 3
        # tMB = Np/q: 100/100=1, 250/80=3.125, 420/60=7
        assert abs(result["material_balance_time"][0] - 1.0) < 1e-4
        assert abs(result["material_balance_time"][1] - 3.125) < 1e-4
        assert abs(result["material_balance_time"][2] - 7.0) < 1e-4

    def test_zero_rate_gives_zero_mbt(self):
        result = json.loads(calculate_material_balance_time(
            [100, 200], [100, 0],
        ))
        assert result["material_balance_time"][1] == 0.0

    def test_mismatched_raises(self):
        with pytest.raises(ValueError, match="same length"):
            calculate_material_balance_time([100], [100, 80])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="not be empty"):
            calculate_material_balance_time([], [])


# ---------------------------------------------------------------------------
# calculate_blasingame_variables
# ---------------------------------------------------------------------------

class TestBlasingame:
    def test_basic_output_structure(self):
        t, rates, cum, pwf, pi = _make_decline_data(n=10)
        result = json.loads(calculate_blasingame_variables(t, rates, cum, pwf, pi))
        assert result["method"] == "Blasingame"
        assert result["num_points"] == 10
        assert len(result["material_balance_time"]) == 10
        assert len(result["normalized_rate"]) == 10
        assert len(result["rate_integral"]) == 10
        assert len(result["rate_integral_derivative"]) == 10

    def test_mbt_increasing(self):
        t, rates, cum, pwf, pi = _make_decline_data(n=10)
        result = json.loads(calculate_blasingame_variables(t, rates, cum, pwf, pi))
        mbt = result["material_balance_time"]
        # MBT should be monotonically increasing for declining rate
        for i in range(1, len(mbt)):
            assert mbt[i] >= mbt[i - 1]

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            calculate_blasingame_variables([1, 2], [100, 80], [100, 180], [2000, 2100], 3000.0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            calculate_blasingame_variables([1, 2, 3], [100, 80], [100, 180, 260], [2000, 2100, 2200], 3000.0)


# ---------------------------------------------------------------------------
# calculate_agarwal_gardner_variables
# ---------------------------------------------------------------------------

class TestAgarwalGardner:
    def test_basic_output_structure(self):
        t, rates, cum, pwf, pi = _make_decline_data(n=10)
        result = json.loads(calculate_agarwal_gardner_variables(t, rates, cum, pwf, pi))
        assert result["method"] == "Agarwal-Gardner"
        assert result["num_points"] == 10
        assert "normalized_rate" in result
        assert "inverse_normalized_rate" in result
        assert "cumulative_normalized" in result

    def test_normalized_rate_values(self):
        # Simple check: q=100, dp=1000 -> qn=0.1
        result = json.loads(calculate_agarwal_gardner_variables(
            [1, 2, 3], [100, 80, 60], [100, 180, 240],
            [2000, 2100, 2200], 3000.0,
        ))
        assert abs(result["normalized_rate"][0] - 0.1) < 1e-4

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            calculate_agarwal_gardner_variables([1], [100], [100], [2000], 3000.0)


# ---------------------------------------------------------------------------
# calculate_npi_variables
# ---------------------------------------------------------------------------

class TestNPI:
    def test_basic_output_structure(self):
        t, rates, _, pwf, pi = _make_decline_data(n=10)
        result = json.loads(calculate_npi_variables(t, rates, pwf, pi))
        assert result["method"] == "NPI (Normalized Pressure Integral)"
        assert result["num_points"] == 10
        assert len(result["npi"]) == 10
        assert len(result["npi_derivative"]) == 10

    def test_npi_first_value_zero(self):
        t, rates, _, pwf, pi = _make_decline_data(n=5)
        result = json.loads(calculate_npi_variables(t, rates, pwf, pi))
        # NPI at first point should be 0 (no integration yet)
        assert result["npi"][0] == 0.0

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            calculate_npi_variables([1, 2], [100, 80], [2000, 2100], 3000.0)

    def test_negative_pi_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_npi_variables([1, 2, 3], [100, 80, 60], [2000, 2100, 2200], -100.0)


# ---------------------------------------------------------------------------
# calculate_flowing_material_balance
# ---------------------------------------------------------------------------

class TestFlowingMaterialBalance:
    def test_basic_output_structure(self):
        # Declining rate with constant pressure -> should get a slope
        rates = [500, 450, 400, 350, 300, 250, 200, 150, 100, 80]
        pwf = [2000] * 10
        result = json.loads(calculate_flowing_material_balance(
            rates, pwf, 3000.0, 1.2, 1e-5,
        ))
        assert result["method"] == "Flowing Material Balance"
        assert "fmb_slope" in result
        assert "fmb_intercept" in result
        assert "r_squared" in result
        assert result["num_valid_points"] == 10

    def test_negative_fvf_raises(self):
        with pytest.raises(ValueError, match="fluid_fvf must be positive"):
            calculate_flowing_material_balance(
                [100, 80, 60], [2000, 2100, 2200], 3000.0, -1.2, 1e-5,
            )

    def test_negative_ct_raises(self):
        with pytest.raises(ValueError, match="total_compressibility must be positive"):
            calculate_flowing_material_balance(
                [100, 80, 60], [2000, 2100, 2200], 3000.0, 1.2, -1e-5,
            )

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            calculate_flowing_material_balance(
                [100, 80], [2000, 2100], 3000.0, 1.2, 1e-5,
            )


# ---------------------------------------------------------------------------
# calculate_sqrt_time_analysis
# ---------------------------------------------------------------------------

class TestSqrtTimeAnalysis:
    def test_basic_linear_flow(self):
        # Simulate linear flow: (Pi-Pwf)/q = m * sqrt(t) + b
        import numpy as np
        n = 20
        times = list(range(1, n + 1))
        slope_true = 0.5
        intercept_true = 2.0
        pi = 3000.0
        pwf_val = 2000.0
        # dp/q = slope * sqrt(t) + intercept  =>  q = dp / (slope*sqrt(t) + intercept)
        dp = pi - pwf_val
        rates = [dp / (slope_true * math.sqrt(t) + intercept_true) for t in times]
        pwf = [pwf_val] * n

        result = json.loads(calculate_sqrt_time_analysis(rates, times, pwf, pi))
        assert result["method"] == "Square Root of Time Analysis"
        assert abs(result["linear_flow_slope"] - slope_true) < 0.05
        assert result["r_squared"] > 0.99

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            calculate_sqrt_time_analysis([100, 80], [1, 2], [2000, 2100], 3000.0)

    def test_negative_pi_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_sqrt_time_analysis(
                [100, 80, 60], [1, 2, 3], [2000, 2100, 2200], -100.0,
            )


# ---------------------------------------------------------------------------
# calculate_rta_permeability
# ---------------------------------------------------------------------------

class TestRTAPermeability:
    def test_sqrt_k_xf_only(self):
        result = json.loads(calculate_rta_permeability(
            slope_from_linear_flow=0.5,
            net_pay_ft=50.0,
            porosity=0.10,
            viscosity_cp=1.0,
            total_compressibility=1e-5,
        ))
        assert "sqrt_k_times_xf" in result
        assert result["sqrt_k_times_xf"] > 0
        assert "note" in result  # No xf provided, so note should be present
        assert "permeability_md" not in result

    def test_with_fracture_half_length(self):
        result = json.loads(calculate_rta_permeability(
            slope_from_linear_flow=0.5,
            net_pay_ft=50.0,
            porosity=0.10,
            viscosity_cp=1.0,
            total_compressibility=1e-5,
            fracture_half_length_ft=300.0,
        ))
        assert "permeability_md" in result
        assert result["permeability_md"] > 0
        assert result["fracture_half_length_ft"] == 300.0
        # Verify consistency: sqrt(k) * xf == sqrt_k_xf
        sqrt_k_xf = result["sqrt_k_times_xf"]
        k = result["permeability_md"]
        assert abs(math.sqrt(k) * 300.0 - sqrt_k_xf) < 0.01

    def test_negative_slope_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_rta_permeability(-0.5, 50, 0.1, 1.0, 1e-5)

    def test_zero_pay_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_rta_permeability(0.5, 0, 0.1, 1.0, 1e-5)

    def test_invalid_porosity_raises(self):
        with pytest.raises(ValueError, match="porosity"):
            calculate_rta_permeability(0.5, 50, 0.0, 1.0, 1e-5)
        with pytest.raises(ValueError, match="porosity"):
            calculate_rta_permeability(0.5, 50, 1.5, 1.0, 1e-5)

    def test_negative_xf_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_rta_permeability(0.5, 50, 0.1, 1.0, 1e-5, fracture_half_length_ft=-100)
