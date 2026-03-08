"""Tests for decline curve analysis tools."""

import json

import numpy as np
import pytest

from petro_mcp.tools.decline import calculate_eur, fit_decline_curve


def _generate_exponential_data(qi=1000, Di=0.05, n=36):
    """Generate synthetic exponential decline data."""
    t = np.arange(n, dtype=float)
    rates = qi * np.exp(-Di * t)
    return [{"time": float(t_), "rate": float(r)} for t_, r in zip(t, rates)]


def _generate_hyperbolic_data(qi=1000, Di=0.05, b=1.2, n=36):
    """Generate synthetic hyperbolic decline data."""
    t = np.arange(n, dtype=float)
    rates = qi / (1 + b * Di * t) ** (1 / b)
    return [{"time": float(t_), "rate": float(r)} for t_, r in zip(t, rates)]


class TestFitDeclineCurve:
    def test_fit_exponential(self):
        data = _generate_exponential_data()
        result = json.loads(fit_decline_curve(data, model="exponential"))
        assert result["model"] == "exponential"
        assert result["r_squared"] > 0.99
        assert abs(result["parameters"]["qi"] - 1000) < 50
        assert abs(result["parameters"]["Di"] - 0.05) < 0.01

    def test_fit_hyperbolic(self):
        data = _generate_hyperbolic_data()
        result = json.loads(fit_decline_curve(data, model="hyperbolic"))
        assert result["model"] == "hyperbolic"
        assert result["r_squared"] > 0.99
        assert 0 <= result["parameters"]["b"] <= 2.0

    def test_fit_harmonic(self):
        data = _generate_hyperbolic_data(b=1.0)
        result = json.loads(fit_decline_curve(data, model="harmonic"))
        assert result["model"] == "harmonic"
        assert result["r_squared"] > 0.95

    def test_invalid_model(self):
        data = _generate_exponential_data()
        with pytest.raises(ValueError, match="Unknown model"):
            fit_decline_curve(data, model="invalid")

    def test_too_few_points(self):
        data = [{"time": 0, "rate": 100}, {"time": 1, "rate": 90}]
        with pytest.raises(ValueError, match="at least 3"):
            fit_decline_curve(data)

    def test_oil_key_instead_of_rate(self):
        data = [{"oil": 1000 * np.exp(-0.05 * t)} for t in range(36)]
        result = json.loads(fit_decline_curve(data, model="exponential"))
        assert result["r_squared"] > 0.99

    def test_b_factor_bounded(self):
        data = _generate_hyperbolic_data(b=1.5)
        result = json.loads(fit_decline_curve(data, model="hyperbolic"))
        assert 0 <= result["parameters"]["b"] <= 2.0

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fit_decline_curve([])

    def test_gas_key(self):
        data = [{"gas": 5000 * np.exp(-0.03 * t)} for t in range(36)]
        result = json.loads(fit_decline_curve(data, model="exponential"))
        assert result["r_squared"] > 0.99


class TestCalculateEUR:
    def test_exponential_eur(self):
        result = json.loads(calculate_eur(qi=500, Di=0.05, model="exponential"))
        assert result["eur"] > 0
        assert result["time_to_economic_limit_months"] > 0

    def test_hyperbolic_eur(self):
        result = json.loads(calculate_eur(qi=1000, Di=0.08, b=1.2, model="hyperbolic"))
        assert result["eur"] > 0
        assert "cumulative_milestones" in result

    def test_harmonic_eur(self):
        result = json.loads(calculate_eur(qi=500, Di=0.03, model="harmonic"))
        assert result["eur"] > 0

    def test_invalid_qi(self):
        with pytest.raises(ValueError, match="qi must be positive"):
            calculate_eur(qi=-100, Di=0.05)

    def test_invalid_di(self):
        with pytest.raises(ValueError, match="Di must be positive"):
            calculate_eur(qi=500, Di=0)

    def test_b_factor_clipped(self):
        result = json.loads(calculate_eur(qi=500, Di=0.05, b=5.0, model="hyperbolic"))
        assert result["parameters"]["b"] == 2.0

    def test_eur_increases_with_qi(self):
        eur_low = json.loads(calculate_eur(qi=200, Di=0.05, b=1.0))["eur"]
        eur_high = json.loads(calculate_eur(qi=1000, Di=0.05, b=1.0))["eur"]
        assert eur_high > eur_low


def _generate_modified_hyperbolic_data(qi=1000, Di=0.08, b=1.2, Dmin=0.005, n=120):
    """Generate synthetic modified hyperbolic decline data."""
    from petro_mcp.tools.decline import _modified_hyperbolic

    t = np.arange(n, dtype=float)
    rates = _modified_hyperbolic(t, qi, Di, b, Dmin)
    return [{"time": float(t_), "rate": float(r)} for t_, r in zip(t, rates)]


def _generate_duong_data(qi=500, a=1.5, m=1.1, n=60):
    """Generate synthetic Duong decline data."""
    from petro_mcp.tools.decline import _duong

    t = np.arange(n, dtype=float)
    rates = _duong(t, qi, a, m)
    return [{"time": float(t_), "rate": float(r)} for t_, r in zip(t, rates)]


class TestModifiedHyperbolic:
    def test_fit_modified_hyperbolic(self):
        """Generate modified hyperbolic data and fit it."""
        data = _generate_modified_hyperbolic_data(qi=1000, Di=0.08, b=1.2, Dmin=0.005, n=120)
        result = json.loads(fit_decline_curve(data, model="modified_hyperbolic"))
        assert result["model"] == "modified_hyperbolic"
        assert result["r_squared"] > 0.95
        # Verify Dmin parameter is recovered (within reasonable tolerance)
        assert 0.0005 < result["parameters"]["Dmin"] < 0.02

    def test_modified_hyp_eur_less_than_pure_hyp(self):
        """Modified hyperbolic EUR should be less than pure hyperbolic EUR."""
        eur_hyp = json.loads(
            calculate_eur(qi=1000, Di=0.05, b=1.2, model="hyperbolic")
        )["eur"]
        eur_mod = json.loads(
            calculate_eur(qi=1000, Di=0.05, b=1.2, Dmin=0.005, model="modified_hyperbolic")
        )["eur"]
        assert eur_mod < eur_hyp

    def test_switch_time_correct(self):
        """Verify the switch from hyperbolic to exponential happens at right time."""
        from petro_mcp.tools.decline import _modified_hyperbolic

        qi, Di, b, Dmin = 1000, 0.08, 1.2, 0.005
        t_switch = (Di - Dmin) / (b * Di * Dmin)
        # Rate should be continuous at switch point
        t = np.array([t_switch - 0.01, t_switch, t_switch + 0.01])
        rates = _modified_hyperbolic(t, qi, Di, b, Dmin)
        # Check continuity: adjacent rates should be very close
        assert abs(rates[0] - rates[1]) / rates[1] < 0.01
        assert abs(rates[1] - rates[2]) / rates[2] < 0.01

    def test_dmin_ge_di_uses_exponential(self):
        """When Dmin >= Di, should behave like exponential."""
        from petro_mcp.tools.decline import _modified_hyperbolic

        t = np.arange(36, dtype=float)
        rates = _modified_hyperbolic(t, 1000, 0.05, 1.0, 0.10)
        expected = 1000 * np.exp(-0.05 * t)
        np.testing.assert_allclose(rates, expected, rtol=1e-10)


class TestDuong:
    def test_fit_duong(self):
        """Generate Duong data and fit it."""
        data = _generate_duong_data(qi=500, a=1.5, m=1.1, n=60)
        result = json.loads(fit_decline_curve(data, model="duong"))
        assert result["model"] == "duong"
        assert result["r_squared"] > 0.95
        # Verify parameters are recovered
        assert abs(result["parameters"]["a"] - 1.5) < 0.5
        assert abs(result["parameters"]["m"] - 1.1) < 0.3

    def test_duong_eur(self):
        """Duong EUR should be positive."""
        result = json.loads(calculate_eur(qi=500, a=1.5, m=1.1, model="duong"))
        assert result["eur"] > 0
        assert result["model"] == "duong"

    def test_duong_decreasing(self):
        """Duong rates should generally decrease over time with appropriate params."""
        from petro_mcp.tools.decline import _duong

        t = np.arange(0, 60, dtype=float)
        # a=1.0, m=1.2 gives clear decline behavior typical of shale wells
        rates = _duong(t, 500, 1.0, 1.2)
        # Check first rate > last rate (overall decline)
        assert rates[0] > rates[-1]
