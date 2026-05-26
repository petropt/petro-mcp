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


