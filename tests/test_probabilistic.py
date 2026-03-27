"""Tests for probabilistic decline curve analysis tools."""

import json

import numpy as np
import pytest

from petro_mcp.tools.probabilistic import (
    bootstrap_decline_parameters,
    calculate_eur_distribution,
    monte_carlo_eur,
    probabilistic_forecast,
    sensitivity_analysis,
)


def _generate_hyperbolic_data(qi=500, Di=0.04, b=1.2, n=48, noise=0.0):
    """Generate synthetic hyperbolic decline data with optional noise."""
    rng = np.random.default_rng(99)
    t = np.arange(n, dtype=float)
    rates = qi / (1 + b * Di * t) ** (1 / b)
    if noise > 0:
        rates = rates + rng.normal(0, noise, n)
        rates = np.maximum(rates, 1.0)
    return [{"time": float(t_), "rate": float(r)} for t_, r in zip(t, rates)]


# -----------------------------------------------------------------------
# Monte Carlo EUR
# -----------------------------------------------------------------------

class TestMonteCarloEur:
    def test_basic_run(self):
        result = json.loads(monte_carlo_eur(500, 100, 0.04, 0.01, num_simulations=500))
        assert result["p10_eur"] > result["p50_eur"] > result["p90_eur"]
        assert result["mean_eur"] > 0
        assert result["num_valid"] > 0

    def test_p10_greater_than_p90(self):
        """P10 (optimistic) should always exceed P90 (conservative)."""
        result = json.loads(monte_carlo_eur(1000, 200, 0.05, 0.01, num_simulations=500))
        assert result["p10_eur"] >= result["p90_eur"]

    def test_zero_std_gives_deterministic(self):
        """With zero standard deviations, all realizations should be identical."""
        result = json.loads(monte_carlo_eur(500, 0, 0.04, 0, b_std=0, num_simulations=100))
        assert result["p10_eur"] == result["p90_eur"]
        assert result["std_eur"] == 0

    def test_lognormal_distribution(self):
        result = json.loads(monte_carlo_eur(500, 100, 0.04, 0.01,
                                            distribution="lognormal", num_simulations=500))
        assert result["distribution"] == "lognormal"

    def test_normal_distribution(self):
        result = json.loads(monte_carlo_eur(500, 100, 0.04, 0.01,
                                            distribution="normal", num_simulations=500))
        assert result["distribution"] == "normal"

    def test_invalid_distribution(self):
        with pytest.raises(ValueError, match="distribution must be"):
            monte_carlo_eur(500, 100, 0.04, 0.01, distribution="uniform")

    def test_negative_qi_mean(self):
        with pytest.raises(ValueError, match="qi_mean must be positive"):
            monte_carlo_eur(-100, 10, 0.04, 0.01)

    def test_negative_di_mean(self):
        with pytest.raises(ValueError, match="di_mean must be positive"):
            monte_carlo_eur(500, 100, -0.04, 0.01)

    def test_negative_std(self):
        with pytest.raises(ValueError, match="non-negative"):
            monte_carlo_eur(500, -10, 0.04, 0.01)

    def test_too_few_simulations(self):
        with pytest.raises(ValueError, match="at least 10"):
            monte_carlo_eur(500, 100, 0.04, 0.01, num_simulations=5)

    def test_confidence_interval(self):
        result = json.loads(monte_carlo_eur(500, 100, 0.04, 0.01, num_simulations=1000))
        ci = result["confidence_interval_80"]
        assert ci["lower"] <= ci["upper"]
        assert ci["lower"] == result["p90_eur"]
        assert ci["upper"] == result["p10_eur"]

    def test_parameter_statistics_present(self):
        result = json.loads(monte_carlo_eur(500, 100, 0.04, 0.01, num_simulations=500))
        for param in ("qi", "di", "b"):
            assert param in result["parameter_statistics"]
            for key in ("p10", "p50", "p90"):
                assert key in result["parameter_statistics"][param]


# -----------------------------------------------------------------------
# Bootstrap decline parameters
# -----------------------------------------------------------------------

class TestBootstrapDecline:
    def test_hyperbolic_bootstrap(self):
        data = _generate_hyperbolic_data(noise=5.0)
        result = json.loads(bootstrap_decline_parameters(data, model="hyperbolic", num_bootstrap=100))
        assert result["model"] == "hyperbolic"
        assert result["num_successful_fits"] > 50
        assert "qi" in result["parameter_statistics"]
        assert "Di" in result["parameter_statistics"]
        assert "b" in result["parameter_statistics"]

    def test_exponential_bootstrap(self):
        t = np.arange(36, dtype=float)
        rates = 500 * np.exp(-0.05 * t) + np.random.default_rng(42).normal(0, 3, 36)
        rates = np.maximum(rates, 1.0)
        data = [{"time": float(t_), "rate": float(r)} for t_, r in zip(t, rates)]
        result = json.loads(bootstrap_decline_parameters(data, model="exponential", num_bootstrap=100))
        assert result["model"] == "exponential"
        assert "qi" in result["parameter_statistics"]
        assert "Di" in result["parameter_statistics"]

    def test_eur_statistics(self):
        data = _generate_hyperbolic_data(noise=5.0)
        result = json.loads(bootstrap_decline_parameters(data, model="hyperbolic", num_bootstrap=100))
        eur = result["eur_statistics"]
        assert eur["p10"] >= eur["p90"]
        assert eur["mean"] > 0

    def test_too_few_points(self):
        data = [{"time": 0, "rate": 100}, {"time": 1, "rate": 90}]
        with pytest.raises(ValueError, match="at least 5"):
            bootstrap_decline_parameters(data)

    def test_invalid_model(self):
        data = _generate_hyperbolic_data()
        with pytest.raises(ValueError, match="Unknown model"):
            bootstrap_decline_parameters(data, model="invalid")


# -----------------------------------------------------------------------
# EUR distribution fitting
# -----------------------------------------------------------------------

class TestEurDistribution:
    def test_lognormal_fit(self):
        rng = np.random.default_rng(42)
        eur_vals = rng.lognormal(mean=12, sigma=0.5, size=200).tolist()
        result = json.loads(calculate_eur_distribution(eur_vals, distribution="lognormal"))
        assert result["distribution"] == "lognormal"
        assert result["p10_eur"] > result["p50_eur"] > result["p90_eur"]
        assert "sigma" in result["distribution_parameters"]
        assert "mu" in result["distribution_parameters"]

    def test_normal_fit(self):
        rng = np.random.default_rng(42)
        eur_vals = rng.normal(100000, 10000, size=200).tolist()
        eur_vals = [max(1, v) for v in eur_vals]
        result = json.loads(calculate_eur_distribution(eur_vals, distribution="normal"))
        assert result["distribution"] == "normal"
        assert "mean" in result["distribution_parameters"]
        assert "std" in result["distribution_parameters"]

    def test_goodness_of_fit(self):
        rng = np.random.default_rng(42)
        eur_vals = rng.lognormal(mean=12, sigma=0.3, size=500).tolist()
        result = json.loads(calculate_eur_distribution(eur_vals))
        gof = result["goodness_of_fit"]
        assert "ks_statistic" in gof
        assert "ks_pvalue" in gof
        assert isinstance(gof["acceptable_fit"], bool)

    def test_sample_statistics(self):
        eur_vals = [100000, 120000, 80000, 150000, 90000]
        result = json.loads(calculate_eur_distribution(eur_vals))
        stats = result["sample_statistics"]
        assert stats["count"] == 5
        assert stats["min"] == 80000
        assert stats["max"] == 150000

    def test_too_few_values(self):
        with pytest.raises(ValueError, match="at least 3"):
            calculate_eur_distribution([100])

    def test_invalid_distribution(self):
        with pytest.raises(ValueError, match="distribution must be"):
            calculate_eur_distribution([100, 200, 300], distribution="weibull")


# -----------------------------------------------------------------------
# Sensitivity analysis
# -----------------------------------------------------------------------

class TestSensitivityAnalysis:
    def test_default_ranges(self):
        result = json.loads(sensitivity_analysis(500, 0.04, 1.2))
        assert result["base_eur"] > 0
        assert len(result["sensitivities"]) == 4  # qi, di, b, economic_limit
        assert result["most_sensitive_parameter"] is not None

    def test_sorted_by_swing(self):
        result = json.loads(sensitivity_analysis(500, 0.04, 1.2))
        swings = [s["swing"] for s in result["sensitivities"]]
        assert swings == sorted(swings, reverse=True)

    def test_custom_ranges(self):
        ranges = {"qi": [300, 700], "di": [0.02, 0.06]}
        result = json.loads(sensitivity_analysis(500, 0.04, 1.2, parameter_ranges=ranges))
        assert len(result["sensitivities"]) == 2
        params = {s["parameter"] for s in result["sensitivities"]}
        assert params == {"qi", "di"}

    def test_negative_qi(self):
        with pytest.raises(ValueError, match="qi must be positive"):
            sensitivity_analysis(-500, 0.04, 1.2)

    def test_negative_di(self):
        with pytest.raises(ValueError, match="di must be positive"):
            sensitivity_analysis(500, -0.04, 1.2)

    def test_pct_impact_present(self):
        result = json.loads(sensitivity_analysis(500, 0.04, 1.2))
        for s in result["sensitivities"]:
            assert "pct_impact" in s
            assert s["pct_impact"] >= 0


# -----------------------------------------------------------------------
# Probabilistic forecast
# -----------------------------------------------------------------------

class TestProbabilisticForecast:
    def test_basic_forecast(self):
        result = json.loads(probabilistic_forecast(
            {"mean": 500, "std": 100},
            {"mean": 0.04, "std": 0.01},
            {"mean": 1.2, "std": 0.3},
            forecast_months=120,
            num_simulations=200,
        ))
        assert result["forecast_months"] == 120
        assert result["num_simulations"] == 200
        assert result["eur_summary"]["p10"] >= result["eur_summary"]["p90"]

    def test_profiles_present(self):
        result = json.loads(probabilistic_forecast(
            {"mean": 500, "std": 100},
            {"mean": 0.04, "std": 0.01},
            {"mean": 1.2, "std": 0.3},
            forecast_months=60,
            num_simulations=100,
        ))
        profiles = result["profiles"]
        assert "months" in profiles
        assert "p10_rates" in profiles
        assert "p50_rates" in profiles
        assert "p90_rates" in profiles
        assert len(profiles["p10_rates"]) == len(profiles["months"])

    def test_p10_above_p90_rates(self):
        """At any time step, P10 rate should be >= P90 rate."""
        result = json.loads(probabilistic_forecast(
            {"mean": 500, "std": 100},
            {"mean": 0.04, "std": 0.01},
            {"mean": 1.2, "std": 0.3},
            forecast_months=60,
            num_simulations=200,
        ))
        profiles = result["profiles"]
        for p10, p90 in zip(profiles["p10_rates"], profiles["p90_rates"]):
            assert p10 >= p90

    def test_cumulative_milestones(self):
        result = json.loads(probabilistic_forecast(
            {"mean": 500, "std": 100},
            {"mean": 0.04, "std": 0.01},
            {"mean": 1.2, "std": 0.3},
            forecast_months=240,
            num_simulations=100,
        ))
        milestones = result["cumulative_milestones"]
        assert "1yr" in milestones
        assert "5yr" in milestones
        for yr_data in milestones.values():
            assert yr_data["p10"] >= yr_data["p90"]

    def test_negative_qi_mean(self):
        with pytest.raises(ValueError, match="qi_dist mean must be positive"):
            probabilistic_forecast(
                {"mean": -500, "std": 100},
                {"mean": 0.04, "std": 0.01},
                {"mean": 1.2, "std": 0.3},
            )

    def test_zero_std_deterministic(self):
        result = json.loads(probabilistic_forecast(
            {"mean": 500, "std": 0},
            {"mean": 0.04, "std": 0},
            {"mean": 1.2, "std": 0},
            forecast_months=60,
            num_simulations=50,
        ))
        profiles = result["profiles"]
        # With zero std, all percentiles should be identical
        for p10, p50, p90 in zip(profiles["p10_rates"], profiles["p50_rates"], profiles["p90_rates"]):
            assert abs(p10 - p90) < 0.01
