"""Tests for advanced decline curve analysis tools (petbox-dca integration)."""

import json

import numpy as np
import petbox.dca as dca
import pytest

from petro_mcp.tools.advanced_decline import (
    DAYS_PER_MONTH,
    fit_duong_decline,
    fit_ple_decline,
    fit_sepd_decline,
    forecast_advanced_decline,
)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def _generate_ple_data(qi=1000, Di=0.002, Dinf=0.0003, n=0.5, n_months=60):
    """Generate synthetic PLE data in monthly time/rate dicts."""
    model = dca.PLE(qi=qi, Di=Di, Dinf=Dinf, n=n)
    t_months = np.arange(n_months, dtype=float)
    t_days = t_months * DAYS_PER_MONTH
    t_days = np.maximum(t_days, 1.0)
    rates = model.rate(t_days)
    return [{"time": float(tm), "rate": float(r)} for tm, r in zip(t_months, rates)]


def _generate_duong_data(qi=500, a=1.5, m=1.1, n_months=60):
    """Generate synthetic Duong data in monthly time/rate dicts."""
    model = dca.Duong(qi=qi, a=a, m=m)
    t_months = np.arange(n_months, dtype=float)
    t_days = t_months * DAYS_PER_MONTH
    t_days = np.maximum(t_days, 1.0)
    rates = model.rate(t_days)
    return [{"time": float(tm), "rate": float(r)} for tm, r in zip(t_months, rates)]


def _generate_sepd_data(qi=1000, tau=500, n=0.5, n_months=60):
    """Generate synthetic SEPD data in monthly time/rate dicts."""
    model = dca.SE(qi=qi, tau=tau, n=n)
    t_months = np.arange(n_months, dtype=float)
    t_days = t_months * DAYS_PER_MONTH
    t_days = np.maximum(t_days, 1.0)
    rates = model.rate(t_days)
    return [{"time": float(tm), "rate": float(r)} for tm, r in zip(t_months, rates)]


# ---------------------------------------------------------------------------
# PLE Tests
# ---------------------------------------------------------------------------

class TestFitPLEDecline:
    def test_fit_ple_basic(self):
        data = _generate_ple_data()
        result = json.loads(fit_ple_decline(data))
        assert result["model"] == "ple"
        assert result["r_squared"] > 0.95
        assert "qi" in result["parameters"]
        assert "Di" in result["parameters"]
        assert "Dinf" in result["parameters"]
        assert "n" in result["parameters"]

    def test_fit_ple_returns_all_keys(self):
        data = _generate_ple_data()
        result = json.loads(fit_ple_decline(data))
        expected_keys = {
            "model", "parameters", "parameter_errors", "r_squared",
            "num_data_points", "predicted_rates", "residuals_summary", "units",
        }
        assert set(result.keys()) == expected_keys

    def test_fit_ple_parameter_recovery(self):
        """Parameters should be approximately recovered from clean synthetic data."""
        data = _generate_ple_data(qi=800, Di=0.003, Dinf=0.0002, n=0.4, n_months=120)
        result = json.loads(fit_ple_decline(data))
        assert result["r_squared"] > 0.99
        assert abs(result["parameters"]["qi"] - 800) / 800 < 0.15

    def test_fit_ple_with_oil_key(self):
        """Accept 'oil' key instead of 'rate'."""
        model = dca.PLE(qi=1000, Di=0.002, Dinf=0.0003, n=0.5)
        t_days = np.maximum(np.arange(36, dtype=float) * DAYS_PER_MONTH, 1.0)
        rates = model.rate(t_days)
        data = [{"oil": float(r)} for r in rates]
        result = json.loads(fit_ple_decline(data))
        assert result["r_squared"] > 0.95

    def test_fit_ple_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fit_ple_decline([])

    def test_fit_ple_too_few_points(self):
        data = [{"time": 0, "rate": 100}, {"time": 1, "rate": 90}]
        with pytest.raises(ValueError, match="at least 3"):
            fit_ple_decline(data)


# ---------------------------------------------------------------------------
# Duong Tests
# ---------------------------------------------------------------------------

class TestFitDuongDecline:
    def test_fit_duong_basic(self):
        data = _generate_duong_data()
        result = json.loads(fit_duong_decline(data))
        assert result["model"] == "duong"
        assert result["r_squared"] > 0.95
        assert "qi" in result["parameters"]
        assert "a" in result["parameters"]
        assert "m" in result["parameters"]

    def test_fit_duong_returns_all_keys(self):
        data = _generate_duong_data()
        result = json.loads(fit_duong_decline(data))
        expected_keys = {
            "model", "parameters", "parameter_errors", "r_squared",
            "num_data_points", "predicted_rates", "residuals_summary", "units",
        }
        assert set(result.keys()) == expected_keys

    def test_fit_duong_parameter_recovery(self):
        data = _generate_duong_data(qi=800, a=2.0, m=1.2, n_months=120)
        result = json.loads(fit_duong_decline(data))
        assert result["r_squared"] > 0.99
        assert abs(result["parameters"]["a"] - 2.0) < 1.0
        assert abs(result["parameters"]["m"] - 1.2) < 0.5

    def test_fit_duong_with_gas_key(self):
        model = dca.Duong(qi=5000, a=1.5, m=1.1)
        t_days = np.maximum(np.arange(36, dtype=float) * DAYS_PER_MONTH, 1.0)
        rates = model.rate(t_days)
        data = [{"gas": float(r)} for r in rates]
        result = json.loads(fit_duong_decline(data))
        assert result["r_squared"] > 0.95

    def test_fit_duong_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fit_duong_decline([])


# ---------------------------------------------------------------------------
# SEPD Tests
# ---------------------------------------------------------------------------

class TestFitSEPDDecline:
    def test_fit_sepd_basic(self):
        data = _generate_sepd_data()
        result = json.loads(fit_sepd_decline(data))
        assert result["model"] == "sepd"
        assert result["r_squared"] > 0.95
        assert "qi" in result["parameters"]
        assert "tau" in result["parameters"]
        assert "n" in result["parameters"]

    def test_fit_sepd_returns_all_keys(self):
        data = _generate_sepd_data()
        result = json.loads(fit_sepd_decline(data))
        expected_keys = {
            "model", "parameters", "parameter_errors", "r_squared",
            "num_data_points", "predicted_rates", "residuals_summary", "units",
        }
        assert set(result.keys()) == expected_keys

    def test_fit_sepd_parameter_recovery(self):
        data = _generate_sepd_data(qi=1200, tau=600, n=0.6, n_months=120)
        result = json.loads(fit_sepd_decline(data))
        assert result["r_squared"] > 0.99
        assert abs(result["parameters"]["qi"] - 1200) / 1200 < 0.15

    def test_fit_sepd_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fit_sepd_decline([])


# ---------------------------------------------------------------------------
# Forecast Tests
# ---------------------------------------------------------------------------

class TestForecastAdvancedDecline:
    def test_forecast_ple(self):
        params = {"qi": 1000, "Di": 0.002, "Dinf": 0.0003, "n": 0.5}
        result = json.loads(forecast_advanced_decline("ple", params))
        assert result["model"] == "ple"
        assert result["eur"] > 0
        assert "cumulative_milestones" in result
        assert "monthly_rates" in result

    def test_forecast_duong(self):
        params = {"qi": 500, "a": 1.5, "m": 1.1}
        result = json.loads(forecast_advanced_decline("duong", params))
        assert result["model"] == "duong"
        assert result["eur"] > 0

    def test_forecast_sepd(self):
        params = {"qi": 1000, "tau": 500, "n": 0.5}
        result = json.loads(forecast_advanced_decline("sepd", params))
        assert result["model"] == "sepd"
        assert result["eur"] > 0

    def test_forecast_thm(self):
        params = {"qi": 1000, "Di": 0.8, "bi": 2.0, "bf": 0.5, "telf": 365}
        result = json.loads(forecast_advanced_decline("thm", params))
        assert result["model"] == "thm"
        assert result["eur"] > 0

    def test_forecast_thm_with_terminal(self):
        params = {
            "qi": 1000, "Di": 0.8, "bi": 2.0, "bf": 0.5,
            "telf": 365, "bterm": 0.05, "tterm": 10.0,
        }
        result = json.loads(forecast_advanced_decline("thm", params))
        assert result["eur"] > 0

    def test_forecast_invalid_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            forecast_advanced_decline("invalid", {"qi": 100})

    def test_forecast_missing_params(self):
        with pytest.raises(ValueError, match="Missing required"):
            forecast_advanced_decline("ple", {"qi": 100})

    def test_forecast_economic_limit(self):
        params = {"qi": 100, "Di": 0.002, "Dinf": 0.0003, "n": 0.5}
        result = json.loads(forecast_advanced_decline("ple", params, economic_limit=50))
        assert result["final_rate"] >= 50 or result["time_to_economic_limit_months"] < 360

    def test_forecast_returns_all_keys(self):
        params = {"qi": 1000, "Di": 0.002, "Dinf": 0.0003, "n": 0.5}
        result = json.loads(forecast_advanced_decline("ple", params))
        expected_keys = {
            "model", "parameters", "forecast_months", "economic_limit",
            "eur", "eur_unit", "time_to_economic_limit_months",
            "time_to_economic_limit_years", "cumulative_milestones",
            "final_rate", "monthly_rates", "units",
        }
        assert set(result.keys()) == expected_keys

    def test_forecast_custom_months(self):
        params = {"qi": 1000, "Di": 0.002, "Dinf": 0.0003, "n": 0.5}
        result = json.loads(forecast_advanced_decline("ple", params, forecast_months=60))
        assert result["forecast_months"] == 60

    def test_forecast_zero_months_raises(self):
        params = {"qi": 1000, "Di": 0.002, "Dinf": 0.0003, "n": 0.5}
        with pytest.raises(ValueError, match="forecast_months must be >= 1"):
            forecast_advanced_decline("ple", params, forecast_months=0)

    def test_forecast_negative_months_raises(self):
        params = {"qi": 1000, "Di": 0.002, "Dinf": 0.0003, "n": 0.5}
        with pytest.raises(ValueError, match="forecast_months must be >= 1"):
            forecast_advanced_decline("ple", params, forecast_months=-5)

    def test_forecast_negative_economic_limit_raises(self):
        params = {"qi": 1000, "Di": 0.002, "Dinf": 0.0003, "n": 0.5}
        with pytest.raises(ValueError, match="economic_limit must be non-negative"):
            forecast_advanced_decline("ple", params, economic_limit=-1.0)


# ---------------------------------------------------------------------------
# Integration: fit then forecast round-trip
# ---------------------------------------------------------------------------

class TestFitThenForecast:
    def test_ple_fit_then_forecast(self):
        data = _generate_ple_data(qi=1000, n_months=60)
        fit_result = json.loads(fit_ple_decline(data))
        params = fit_result["parameters"]
        forecast_result = json.loads(forecast_advanced_decline("ple", params))
        assert forecast_result["eur"] > 0

    def test_duong_fit_then_forecast(self):
        data = _generate_duong_data(qi=500, n_months=60)
        fit_result = json.loads(fit_duong_decline(data))
        params = fit_result["parameters"]
        forecast_result = json.loads(forecast_advanced_decline("duong", params))
        assert forecast_result["eur"] > 0

    def test_sepd_fit_then_forecast(self):
        data = _generate_sepd_data(qi=1000, n_months=60)
        fit_result = json.loads(fit_sepd_decline(data))
        params = fit_result["parameters"]
        forecast_result = json.loads(forecast_advanced_decline("sepd", params))
        assert forecast_result["eur"] > 0
