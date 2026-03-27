"""Probabilistic decline curve analysis for reserve estimation.

Provides Monte Carlo simulation, bootstrapping, distribution fitting,
sensitivity analysis, and probabilistic forecasting to generate P10/P50/P90
EUR estimates required for SEC reserve reports.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


# --- Arps decline helpers (standalone to avoid circular imports) ---

def _hyperbolic(t: np.ndarray, qi: float, Di: float, b: float) -> np.ndarray:
    """Hyperbolic decline: q(t) = qi / (1 + b * Di * t)^(1/b)."""
    b = np.clip(b, 0.001, 2.0)
    return qi / (1 + b * Di * t) ** (1 / b)


def _exponential(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
    """Exponential decline: q(t) = qi * exp(-Di * t)."""
    return qi * np.exp(-Di * t)


def _harmonic(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
    """Harmonic decline (b=1): q(t) = qi / (1 + Di * t)."""
    return qi / (1 + Di * t)


_MODELS = {
    "exponential": (_exponential, ["qi", "Di"], [1.0, 0.01], ([0, 0], [np.inf, 10])),
    "hyperbolic": (_hyperbolic, ["qi", "Di", "b"], [1.0, 0.01, 1.0], ([0, 0, 0], [np.inf, 10, 2.0])),
    "harmonic": (_harmonic, ["qi", "Di"], [1.0, 0.01], ([0, 0], [np.inf, 10])),
}

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _compute_eur(qi: float, Di: float, b: float, economic_limit: float,
                 max_time: float = 600) -> float:
    """Compute EUR (bbl) for a single set of decline parameters."""
    days_per_month = 30.44
    t = np.arange(0, max_time + 1, dtype=float)
    b_clipped = np.clip(b, 0.001, 2.0)
    rates = qi / (1 + b_clipped * Di * t) ** (1 / b_clipped)
    rates = np.maximum(rates, 0.0)

    below = np.where(rates < economic_limit)[0]
    if len(below) > 0:
        idx = below[0]
        rates = rates[:idx + 1]
        t = t[:idx + 1]

    if len(t) < 2:
        return 0.0

    return float(_trapz(rates, t)) * days_per_month


def _sample_parameters(mean: float, std: float, n: int,
                       distribution: str, rng: np.random.Generator,
                       lower_clip: float = 0.0) -> np.ndarray:
    """Sample from lognormal or normal distribution with positive clipping."""
    if std <= 0:
        return np.full(n, mean)

    if distribution == "lognormal":
        # Parameterize lognormal from desired mean/std
        variance = std ** 2
        mu = np.log(mean ** 2 / np.sqrt(variance + mean ** 2))
        sigma = np.sqrt(np.log(1 + variance / mean ** 2))
        samples = rng.lognormal(mu, sigma, n)
    else:
        samples = rng.normal(mean, std, n)

    return np.clip(samples, lower_clip, None)


# ---------------------------------------------------------------------------
# Tool 1: Monte Carlo EUR
# ---------------------------------------------------------------------------

def monte_carlo_eur(
    qi_mean: float,
    qi_std: float,
    di_mean: float,
    di_std: float,
    b_mean: float = 1.0,
    b_std: float = 0.3,
    economic_limit: float = 5.0,
    num_simulations: int = 10000,
    distribution: str = "lognormal",
) -> str:
    """Run Monte Carlo simulation to estimate P10/P50/P90 EUR.

    Samples qi, Di, and b from specified distributions, computes EUR for
    each realization, and returns percentile statistics.

    Args:
        qi_mean: Mean initial rate (bbl/day or Mcf/day).
        qi_std: Standard deviation of initial rate.
        di_mean: Mean initial decline rate (1/month).
        di_std: Standard deviation of decline rate.
        b_mean: Mean b-factor (default 1.0).
        b_std: Standard deviation of b-factor (default 0.3).
        economic_limit: Minimum economic rate (default 5.0).
        num_simulations: Number of Monte Carlo realizations (default 10000).
        distribution: Sampling distribution - 'lognormal' or 'normal'.

    Returns:
        JSON string with P10/P50/P90/mean EUR and confidence intervals.
    """
    if qi_mean <= 0:
        raise ValueError("qi_mean must be positive")
    if di_mean <= 0:
        raise ValueError("di_mean must be positive")
    if qi_std < 0 or di_std < 0 or b_std < 0:
        raise ValueError("Standard deviations must be non-negative")
    if num_simulations < 10:
        raise ValueError("num_simulations must be at least 10")
    if distribution not in ("lognormal", "normal"):
        raise ValueError("distribution must be 'lognormal' or 'normal'")

    # Cap simulations for performance
    num_simulations = min(num_simulations, 100000)

    rng = np.random.default_rng(42)

    qi_samples = _sample_parameters(qi_mean, qi_std, num_simulations, distribution, rng, lower_clip=1.0)
    di_samples = _sample_parameters(di_mean, di_std, num_simulations, distribution, rng, lower_clip=0.0001)
    b_samples = _sample_parameters(b_mean, b_std, num_simulations, "normal", rng, lower_clip=0.001)
    b_samples = np.clip(b_samples, 0.001, 2.0)

    eur_values = np.array([
        _compute_eur(qi, di, b, economic_limit)
        for qi, di, b in zip(qi_samples, di_samples, b_samples)
    ])

    # Filter out zero EURs (non-convergent cases)
    valid = eur_values > 0
    eur_valid = eur_values[valid]

    if len(eur_valid) < 3:
        raise ValueError("Too few valid simulations — check input parameters")

    p10 = float(np.percentile(eur_valid, 90))   # P10 = 90th percentile (10% chance of exceeding)
    p50 = float(np.percentile(eur_valid, 50))
    p90 = float(np.percentile(eur_valid, 10))   # P90 = 10th percentile (90% chance of exceeding)
    mean_eur = float(np.mean(eur_valid))

    result: dict[str, Any] = {
        "p10_eur": round(p10, 0),
        "p50_eur": round(p50, 0),
        "p90_eur": round(p90, 0),
        "mean_eur": round(mean_eur, 0),
        "std_eur": round(float(np.std(eur_valid)), 0),
        "confidence_interval_80": {
            "lower": round(float(np.percentile(eur_valid, 10)), 0),
            "upper": round(float(np.percentile(eur_valid, 90)), 0),
        },
        "num_simulations": num_simulations,
        "num_valid": int(np.sum(valid)),
        "distribution": distribution,
        "input_parameters": {
            "qi": {"mean": qi_mean, "std": qi_std},
            "di": {"mean": di_mean, "std": di_std},
            "b": {"mean": b_mean, "std": b_std},
        },
        "parameter_statistics": {
            "qi": {"p10": round(float(np.percentile(qi_samples, 90)), 2),
                   "p50": round(float(np.percentile(qi_samples, 50)), 2),
                   "p90": round(float(np.percentile(qi_samples, 10)), 2)},
            "di": {"p10": round(float(np.percentile(di_samples, 90)), 6),
                   "p50": round(float(np.percentile(di_samples, 50)), 6),
                   "p90": round(float(np.percentile(di_samples, 10)), 6)},
            "b": {"p10": round(float(np.percentile(b_samples, 90)), 4),
                  "p50": round(float(np.percentile(b_samples, 50)), 4),
                  "p90": round(float(np.percentile(b_samples, 10)), 4)},
        },
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 2: Bootstrap decline parameters
# ---------------------------------------------------------------------------

def bootstrap_decline_parameters(
    production_data: list[dict[str, float]],
    model: str = "hyperbolic",
    num_bootstrap: int = 1000,
) -> str:
    """Resample production data with replacement and refit decline curves.

    Returns confidence intervals on fitted parameters (qi, Di, b) and EUR.

    Args:
        production_data: List of dicts with 'time' and 'rate' keys.
        model: Decline model - 'exponential', 'hyperbolic', or 'harmonic'.
        num_bootstrap: Number of bootstrap iterations (default 1000).

    Returns:
        JSON string with parameter confidence intervals and EUR distribution.
    """
    if model not in _MODELS:
        raise ValueError(f"Unknown model: {model}. Must be one of: {list(_MODELS.keys())}")
    if not production_data or len(production_data) < 5:
        raise ValueError("Need at least 5 data points for bootstrapping")
    if num_bootstrap < 10:
        raise ValueError("num_bootstrap must be at least 10")

    num_bootstrap = min(num_bootstrap, 10000)

    # Extract arrays
    if "time" in production_data[0]:
        t_all = np.array([d["time"] for d in production_data], dtype=float)
        q_all = np.array([d["rate"] for d in production_data], dtype=float)
    else:
        t_all = np.arange(len(production_data), dtype=float)
        for key in ("oil", "gas", "rate"):
            if key in production_data[0]:
                q_all = np.array([d[key] for d in production_data], dtype=float)
                break
        else:
            raise ValueError("Production data must have 'rate', 'oil', or 'gas' key")

    valid = q_all > 0
    t_all = t_all[valid]
    q_all = q_all[valid]
    if len(t_all) < 5:
        raise ValueError("Need at least 5 non-zero data points for bootstrapping")

    func, param_names, p0_template, bounds = _MODELS[model]
    n = len(t_all)

    rng = np.random.default_rng(42)
    fitted_params = []
    eur_values = []

    for _ in range(num_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        idx = np.sort(idx)
        t_boot = t_all[idx]
        q_boot = q_all[idx]

        p0 = list(p0_template)
        p0[0] = float(q_boot[0])
        if len(p0) > 1 and t_boot[-1] > t_boot[0]:
            p0[1] = max(0.001, float((q_boot[0] - q_boot[-1]) / (q_boot[0] * (t_boot[-1] - t_boot[0] + 1))))

        try:
            popt, _ = curve_fit(func, t_boot, q_boot, p0=p0, bounds=bounds, maxfev=5000)
            fitted_params.append(popt)

            # Compute EUR for this realization
            if model == "hyperbolic":
                eur = _compute_eur(popt[0], popt[1], popt[2], 5.0)
            elif model == "exponential":
                eur = _compute_eur(popt[0], popt[1], 0.001, 5.0)
            else:  # harmonic
                eur = _compute_eur(popt[0], popt[1], 1.0, 5.0)
            eur_values.append(eur)
        except (RuntimeError, ValueError):
            continue

    if len(fitted_params) < 3:
        raise ValueError("Bootstrap fitting failed — too few successful fits")

    params_array = np.array(fitted_params)
    eur_array = np.array(eur_values)

    param_stats = {}
    for i, name in enumerate(param_names):
        col = params_array[:, i]
        param_stats[name] = {
            "mean": round(float(np.mean(col)), 6),
            "std": round(float(np.std(col)), 6),
            "p10": round(float(np.percentile(col, 90)), 6),
            "p50": round(float(np.percentile(col, 50)), 6),
            "p90": round(float(np.percentile(col, 10)), 6),
            "ci_90": [round(float(np.percentile(col, 5)), 6),
                      round(float(np.percentile(col, 95)), 6)],
        }

    result: dict[str, Any] = {
        "model": model,
        "num_bootstrap": num_bootstrap,
        "num_successful_fits": len(fitted_params),
        "parameter_statistics": param_stats,
        "eur_statistics": {
            "p10": round(float(np.percentile(eur_array, 90)), 0),
            "p50": round(float(np.percentile(eur_array, 50)), 0),
            "p90": round(float(np.percentile(eur_array, 10)), 0),
            "mean": round(float(np.mean(eur_array)), 0),
            "std": round(float(np.std(eur_array)), 0),
        },
        "num_data_points": n,
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 3: EUR distribution fitting
# ---------------------------------------------------------------------------

def calculate_eur_distribution(
    eur_values: list[float],
    distribution: str = "lognormal",
) -> str:
    """Fit a statistical distribution to EUR values and return percentiles.

    Args:
        eur_values: List of EUR values (e.g., from Monte Carlo or analog wells).
        distribution: Distribution to fit - 'lognormal' or 'normal'.

    Returns:
        JSON string with P10/P50/P90, distribution parameters, and goodness of fit.
    """
    if not eur_values or len(eur_values) < 3:
        raise ValueError("Need at least 3 EUR values")
    if distribution not in ("lognormal", "normal"):
        raise ValueError("distribution must be 'lognormal' or 'normal'")

    arr = np.array(eur_values, dtype=float)
    arr = arr[arr > 0]
    if len(arr) < 3:
        raise ValueError("Need at least 3 positive EUR values")

    if distribution == "lognormal":
        shape, loc, scale = stats.lognorm.fit(arr, floc=0)
        fitted_dist = stats.lognorm(shape, loc, scale)
        dist_params = {
            "sigma": round(float(shape), 6),
            "mu": round(float(np.log(scale)), 6),
            "scale": round(float(scale), 2),
        }
    else:
        loc, scale = stats.norm.fit(arr)
        fitted_dist = stats.norm(loc, scale)
        dist_params = {
            "mean": round(float(loc), 2),
            "std": round(float(scale), 2),
        }

    p10 = float(fitted_dist.ppf(0.90))
    p50 = float(fitted_dist.ppf(0.50))
    p90 = float(fitted_dist.ppf(0.10))

    # Kolmogorov-Smirnov goodness of fit
    ks_stat, ks_pvalue = stats.kstest(arr, fitted_dist.cdf)

    result: dict[str, Any] = {
        "distribution": distribution,
        "p10_eur": round(p10, 0),
        "p50_eur": round(p50, 0),
        "p90_eur": round(p90, 0),
        "mean_eur": round(float(np.mean(arr)), 0),
        "distribution_parameters": dist_params,
        "goodness_of_fit": {
            "ks_statistic": round(float(ks_stat), 6),
            "ks_pvalue": round(float(ks_pvalue), 6),
            "acceptable_fit": bool(ks_pvalue > 0.05),
        },
        "sample_statistics": {
            "count": len(arr),
            "min": round(float(np.min(arr)), 0),
            "max": round(float(np.max(arr)), 0),
            "mean": round(float(np.mean(arr)), 0),
            "std": round(float(np.std(arr)), 0),
            "skewness": round(float(stats.skew(arr)), 4),
            "kurtosis": round(float(stats.kurtosis(arr)), 4),
        },
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 4: Sensitivity analysis (tornado chart data)
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    qi: float,
    di: float,
    b: float,
    economic_limit: float = 5.0,
    parameter_ranges: dict[str, list[float]] | None = None,
) -> str:
    """Vary each decline parameter and compute EUR impact (tornado chart data).

    Args:
        qi: Base initial rate (bbl/day or Mcf/day).
        di: Base initial decline rate (1/month).
        b: Base b-factor.
        economic_limit: Minimum economic rate (default 5.0).
        parameter_ranges: Optional dict mapping parameter name to [low, high].
            Defaults to +/-20% of base values.

    Returns:
        JSON string with base EUR and parameter sensitivity data for tornado plot.
    """
    if qi <= 0:
        raise ValueError("qi must be positive")
    if di <= 0:
        raise ValueError("di must be positive")

    base_eur = _compute_eur(qi, di, b, economic_limit)

    if parameter_ranges is None:
        parameter_ranges = {
            "qi": [qi * 0.8, qi * 1.2],
            "di": [di * 0.8, di * 1.2],
            "b": [max(0.001, b * 0.8), min(2.0, b * 1.2)],
            "economic_limit": [economic_limit * 0.5, economic_limit * 2.0],
        }

    sensitivities = []
    for param, (low, high) in parameter_ranges.items():
        if param == "qi":
            eur_low = _compute_eur(low, di, b, economic_limit)
            eur_high = _compute_eur(high, di, b, economic_limit)
        elif param == "di":
            eur_low = _compute_eur(qi, low, b, economic_limit)
            eur_high = _compute_eur(qi, high, b, economic_limit)
        elif param == "b":
            eur_low = _compute_eur(qi, di, low, economic_limit)
            eur_high = _compute_eur(qi, di, high, economic_limit)
        elif param == "economic_limit":
            eur_low = _compute_eur(qi, di, b, low)
            eur_high = _compute_eur(qi, di, b, high)
        else:
            continue

        swing = abs(eur_high - eur_low)
        sensitivities.append({
            "parameter": param,
            "low_value": round(low, 6),
            "high_value": round(high, 6),
            "eur_at_low": round(eur_low, 0),
            "eur_at_high": round(eur_high, 0),
            "swing": round(swing, 0),
            "pct_impact": round(swing / base_eur * 100, 2) if base_eur > 0 else 0.0,
        })

    # Sort by swing descending (tornado chart order)
    sensitivities.sort(key=lambda x: x["swing"], reverse=True)

    result: dict[str, Any] = {
        "base_eur": round(base_eur, 0),
        "base_parameters": {
            "qi": qi,
            "di": di,
            "b": b,
            "economic_limit": economic_limit,
        },
        "sensitivities": sensitivities,
        "most_sensitive_parameter": sensitivities[0]["parameter"] if sensitivities else None,
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 5: Probabilistic forecast (full P10/P50/P90 rate-time curves)
# ---------------------------------------------------------------------------

def probabilistic_forecast(
    qi_dist: dict[str, float],
    di_dist: dict[str, float],
    b_dist: dict[str, float],
    forecast_months: int = 360,
    economic_limit: float = 5.0,
    num_simulations: int = 1000,
) -> str:
    """Generate P10/P50/P90 rate-time forecast profiles.

    Unlike monte_carlo_eur which returns only EUR summaries, this function
    returns the full rate-time curves at each percentile.

    Args:
        qi_dist: Dict with 'mean' and 'std' for initial rate.
        di_dist: Dict with 'mean' and 'std' for decline rate.
        b_dist: Dict with 'mean' and 'std' for b-factor.
        forecast_months: Forecast duration in months (default 360).
        economic_limit: Minimum economic rate (default 5.0).
        num_simulations: Number of realizations (default 1000).

    Returns:
        JSON string with P10/P50/P90 rate-time and cumulative profiles.
    """
    if qi_dist.get("mean", 0) <= 0:
        raise ValueError("qi_dist mean must be positive")
    if di_dist.get("mean", 0) <= 0:
        raise ValueError("di_dist mean must be positive")
    if forecast_months < 1:
        raise ValueError("forecast_months must be at least 1")

    num_simulations = min(num_simulations, 50000)
    forecast_months = min(forecast_months, 600)

    rng = np.random.default_rng(42)

    qi_mean, qi_std = qi_dist["mean"], qi_dist.get("std", 0)
    di_mean, di_std = di_dist["mean"], di_dist.get("std", 0)
    b_mean, b_std = b_dist["mean"], b_dist.get("std", 0)

    qi_samples = _sample_parameters(qi_mean, qi_std, num_simulations, "lognormal", rng, lower_clip=1.0)
    di_samples = _sample_parameters(di_mean, di_std, num_simulations, "lognormal", rng, lower_clip=0.0001)
    b_samples = _sample_parameters(b_mean, b_std, num_simulations, "normal", rng, lower_clip=0.001)
    b_samples = np.clip(b_samples, 0.001, 2.0)

    t = np.arange(0, forecast_months + 1, dtype=float)
    days_per_month = 30.44

    # Compute all rate profiles
    all_rates = np.zeros((num_simulations, len(t)))
    eur_values = np.zeros(num_simulations)

    for i in range(num_simulations):
        b_clipped = float(np.clip(b_samples[i], 0.001, 2.0))
        rates = qi_samples[i] / (1 + b_clipped * di_samples[i] * t) ** (1 / b_clipped)
        rates = np.maximum(rates, 0.0)

        below = np.where(rates < economic_limit)[0]
        if len(below) > 0:
            rates[below[0]:] = 0.0

        all_rates[i] = rates
        nonzero = rates > 0
        if np.sum(nonzero) >= 2:
            eur_values[i] = float(_trapz(rates[nonzero], t[nonzero])) * days_per_month

    # Percentile profiles at each time step
    p10_rates = np.percentile(all_rates, 90, axis=0)
    p50_rates = np.percentile(all_rates, 50, axis=0)
    p90_rates = np.percentile(all_rates, 10, axis=0)

    # Downsample for output (every 3 months for first 5 years, then yearly)
    output_months = list(range(0, min(60, forecast_months + 1), 3))
    output_months += list(range(60, forecast_months + 1, 12))
    output_months = sorted(set(m for m in output_months if m <= forecast_months))

    profiles = {
        "months": output_months,
        "p10_rates": [round(float(p10_rates[m]), 2) for m in output_months],
        "p50_rates": [round(float(p50_rates[m]), 2) for m in output_months],
        "p90_rates": [round(float(p90_rates[m]), 2) for m in output_months],
    }

    # Cumulative at key intervals
    cum_p10 = np.cumsum(p10_rates) * days_per_month
    cum_p50 = np.cumsum(p50_rates) * days_per_month
    cum_p90 = np.cumsum(p90_rates) * days_per_month

    cumulative_milestones = {}
    for yr in [1, 3, 5, 10, 20, 30]:
        mo = yr * 12
        if mo < len(cum_p10):
            cumulative_milestones[f"{yr}yr"] = {
                "p10": round(float(cum_p10[mo]), 0),
                "p50": round(float(cum_p50[mo]), 0),
                "p90": round(float(cum_p90[mo]), 0),
            }

    valid_eur = eur_values[eur_values > 0]

    result: dict[str, Any] = {
        "forecast_months": forecast_months,
        "num_simulations": num_simulations,
        "eur_summary": {
            "p10": round(float(np.percentile(valid_eur, 90)), 0) if len(valid_eur) > 0 else 0,
            "p50": round(float(np.percentile(valid_eur, 50)), 0) if len(valid_eur) > 0 else 0,
            "p90": round(float(np.percentile(valid_eur, 10)), 0) if len(valid_eur) > 0 else 0,
            "mean": round(float(np.mean(valid_eur)), 0) if len(valid_eur) > 0 else 0,
        },
        "profiles": profiles,
        "cumulative_milestones": cumulative_milestones,
        "input_distributions": {
            "qi": {"mean": qi_mean, "std": qi_std},
            "di": {"mean": di_mean, "std": di_std},
            "b": {"mean": b_mean, "std": b_std},
        },
    }
    return json.dumps(result, indent=2)
