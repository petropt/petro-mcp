"""Arps decline curve analysis.

Implements the three classical Arps models with physics-constrained fits:
    - Exponential (b = 0)
    - Hyperbolic (0 < b < 2)
    - Harmonic   (b = 1)

b-factor is bounded to [0, 2]; non-negative rates are enforced.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from scipy.optimize import curve_fit


def _exponential(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
    return qi * np.exp(-Di * t)


def _hyperbolic(t: np.ndarray, qi: float, Di: float, b: float) -> np.ndarray:
    b = np.clip(b, 0.001, 2.0)
    return qi / (1 + b * Di * t) ** (1 / b)


def _harmonic(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
    return qi / (1 + Di * t)


_MODELS = {
    "exponential": (_exponential, ["qi", "Di"], [1.0, 0.01], ([0, 0], [np.inf, 10])),
    "hyperbolic":  (_hyperbolic,  ["qi", "Di", "b"], [1.0, 0.01, 1.0], ([0, 0, 0], [np.inf, 10, 2.0])),
    "harmonic":    (_harmonic,    ["qi", "Di"], [1.0, 0.01], ([0, 0], [np.inf, 10])),
}


def fit_decline_curve(
    production_data: list[dict[str, float]],
    model: str = "hyperbolic",
) -> str:
    """Fit an Arps decline curve to production data.

    Args:
        production_data: List of dicts with 'time' (months) and 'rate' keys,
            or 'oil'/'gas' keys (time assumed sequential months).
        model: Arps model - 'exponential', 'hyperbolic', or 'harmonic'.

    Returns:
        JSON string with fitted parameters, R-squared, and predicted rates.
    """
    if model not in _MODELS:
        raise ValueError(f"Unknown model: {model}. Must be one of: {list(_MODELS.keys())}")
    if not production_data:
        raise ValueError("production_data is empty -- provide at least 3 data points")

    if "time" in production_data[0]:
        t = np.array([d["time"] for d in production_data], dtype=float)
        q = np.array([d["rate"] for d in production_data], dtype=float)
    else:
        t = np.arange(len(production_data), dtype=float)
        for key in ("oil", "gas", "rate"):
            if key in production_data[0]:
                q = np.array([d[key] for d in production_data], dtype=float)
                break
        else:
            raise ValueError("Production data must have 'rate', 'oil', or 'gas' key")

    valid = q > 0
    t = t[valid]
    q = q[valid]
    if len(t) < 3:
        raise ValueError("Need at least 3 non-zero data points for curve fitting")

    func, param_names, p0_template, bounds = _MODELS[model]
    p0 = list(p0_template)
    p0[0] = float(q[0])
    p0[1] = max(0.001, float((q[0] - q[-1]) / (q[0] * (t[-1] - t[0] + 1))))

    try:
        popt, pcov = curve_fit(func, t, q, p0=p0, bounds=bounds, maxfev=10000)
    except RuntimeError as e:
        raise ValueError(f"Curve fitting failed to converge: {e}") from e

    q_pred = func(t, *popt)
    ss_res = np.sum((q - q_pred) ** 2)
    ss_tot = np.sum((q - np.mean(q)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    params = dict(zip(param_names, [round(float(p), 6) for p in popt]))
    perr = np.sqrt(np.diag(pcov))
    param_errors = dict(zip(param_names, [round(float(e), 6) for e in perr]))

    result: dict[str, Any] = {
        "model": model,
        "parameters": params,
        "parameter_errors": param_errors,
        "r_squared": round(float(r_squared), 6),
        "num_data_points": len(t),
        "predicted_rates": [round(float(v), 2) for v in q_pred],
        "residuals_summary": {
            "mean": round(float(np.mean(q - q_pred)), 2),
            "std": round(float(np.std(q - q_pred)), 2),
            "max_abs": round(float(np.max(np.abs(q - q_pred))), 2),
        },
    }
    return json.dumps(result, indent=2)


def calculate_eur(
    qi: float,
    Di: float = 0.0,
    b: float = 0.0,
    economic_limit: float = 5.0,
    model: str = "hyperbolic",
    max_time: float = 600,
) -> str:
    """Calculate Estimated Ultimate Recovery from Arps decline parameters.

    Args:
        qi: Initial production rate (bbl/day or Mcf/day).
        Di: Initial decline rate (1/month, nominal).
        b: Arps b-factor (0 = exponential, 1 = harmonic, 0-2 = hyperbolic).
        economic_limit: Minimum economic rate (same units as qi).
        model: Arps model - 'exponential', 'hyperbolic', or 'harmonic'.
        max_time: Maximum time in months to integrate.

    Returns:
        JSON string with EUR, time to economic limit, and cumulative profile.
    """
    if qi <= 0:
        raise ValueError("qi must be positive")
    if model not in _MODELS:
        raise ValueError(f"Unknown model: {model}. Must be one of: {list(_MODELS.keys())}")
    if Di <= 0:
        raise ValueError("Di must be positive")

    func, param_names, _, _ = _MODELS[model]
    available = {"qi": qi, "Di": Di, "b": float(np.clip(b, 0.001, 2.0))}
    model_params = [available[p] for p in param_names]

    t = np.arange(0, max_time + 1, dtype=float)
    rates = np.maximum(func(t, *model_params), 0.0)

    below_limit = np.where(rates < economic_limit)[0]
    if len(below_limit) > 0:
        econ_time = int(below_limit[0])
        rates = rates[:econ_time + 1]
        t = t[:econ_time + 1]
    else:
        econ_time = int(max_time)

    days_per_month = 30.44
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    eur_bbl = float(_trapz(rates, t)) * days_per_month

    milestones = {}
    for yr in [1, 3, 5, 10, 20]:
        mo = yr * 12
        if mo <= len(rates):
            cum = float(_trapz(rates[:mo + 1], t[:mo + 1])) * days_per_month
            milestones[f"cum_{yr}yr"] = round(cum, 0)

    out_params = {p: available[p] for p in param_names}
    return json.dumps({
        "model": model,
        "parameters": out_params,
        "economic_limit": economic_limit,
        "eur": round(eur_bbl, 0),
        "eur_unit": "bbl (or Mcf if gas)",
        "time_to_economic_limit_months": econ_time,
        "time_to_economic_limit_years": round(econ_time / 12, 1),
        "cumulative_milestones": milestones,
        "final_rate": round(float(rates[-1]), 2) if len(rates) > 0 else 0,
    }, indent=2)
