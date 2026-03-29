"""Decline curve analysis tools for the petro-mcp server.

Implements Arps decline models with physics-constrained constraints:
- b-factor bounded to [0, 2]
- Terminal decline rate enforcement
- Non-negative rate enforcement
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from scipy.optimize import curve_fit


# --- Arps Decline Models ---

def _exponential(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
    """Exponential decline: q(t) = qi * exp(-Di * t)"""
    return qi * np.exp(-Di * t)


def _hyperbolic(t: np.ndarray, qi: float, Di: float, b: float) -> np.ndarray:
    """Hyperbolic decline: q(t) = qi / (1 + b * Di * t)^(1/b)"""
    b = np.clip(b, 0.001, 2.0)
    return qi / (1 + b * Di * t) ** (1 / b)


def _harmonic(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
    """Harmonic decline (b=1): q(t) = qi / (1 + Di * t)"""
    return qi / (1 + Di * t)


def _modified_hyperbolic(t: np.ndarray, qi: float, Di: float, b: float, Dmin: float) -> np.ndarray:
    """Modified hyperbolic: switches to exponential when D(t) = Dmin.

    q(t) = qi / (1 + b*Di*t)^(1/b)          for t < t_switch
    q(t) = q_switch * exp(-Dmin*(t-t_switch)) for t >= t_switch

    where t_switch = (Di - Dmin) / (b * Di * Dmin)
    """
    b = np.clip(b, 0.001, 2.0)
    t = np.asarray(t, dtype=float)

    # If Dmin >= Di, the decline is already below Dmin from the start; use pure exponential
    if Dmin >= Di:
        return qi * np.exp(-Di * t)

    t_switch = (Di - Dmin) / (b * Di * Dmin)

    if t_switch < 0:
        # No switch needed, pure hyperbolic
        return qi / (1 + b * Di * t) ** (1 / b)

    q_switch = qi / (1 + b * Di * t_switch) ** (1 / b)

    q = np.empty_like(t)
    hyp_mask = t < t_switch
    exp_mask = ~hyp_mask
    q[hyp_mask] = qi / (1 + b * Di * t[hyp_mask]) ** (1 / b)
    q[exp_mask] = q_switch * np.exp(-Dmin * (t[exp_mask] - t_switch))
    return q


def _duong(t: np.ndarray, qi: float, a: float, m: float) -> np.ndarray:
    """Duong decline: q(t) = qi * t^(-m) * exp(a/(1-m) * (t^(1-m) - 1))

    Parameters:
        qi: initial rate (at t=1)
        a: intercept parameter (typically 0.5-2.0)
        m: slope parameter (typically 1.0-1.5)

    Reference: Duong, A.N. (2011). "Rate-Decline Analysis for Fracture-
    Dominated Shale Reservoirs." SPE Reservoir Evaluation & Engineering.
    """
    t = np.asarray(t, dtype=float)
    # Duong model requires t >= 1; shift if t starts at 0
    if len(t) > 0 and t[0] < 0.5:
        t = t + 1.0
    return qi * t ** (-m) * np.exp(a / (1 - m) * (t ** (1 - m) - 1))


_MODELS = {
    "exponential": (_exponential, ["qi", "Di"], [1.0, 0.01], ([0, 0], [np.inf, 10])),
    "hyperbolic": (_hyperbolic, ["qi", "Di", "b"], [1.0, 0.01, 1.0], ([0, 0, 0], [np.inf, 10, 2.0])),
    "harmonic": (_harmonic, ["qi", "Di"], [1.0, 0.01], ([0, 0], [np.inf, 10])),
    "modified_hyperbolic": (
        _modified_hyperbolic,
        ["qi", "Di", "b", "Dmin"],
        [1.0, 0.01, 1.0, 0.001],
        ([0, 0, 0.001, 0], [np.inf, 10, 2.0, 1.0]),
    ),
    "duong": (_duong, ["qi", "a", "m"], [1.0, 1.0, 1.1], ([0, 0.1, 0.5], [np.inf, 5.0, 2.0])),
}


def fit_decline_curve(
    production_data: list[dict[str, float]],
    model: str = "hyperbolic",
) -> str:
    """Fit an Arps decline curve to production data.

    Args:
        production_data: List of dicts with 'time' (months from first production)
            and 'rate' (oil/gas rate) keys. Alternatively, 'date' and 'oil'/'gas' keys.
        model: Decline model - 'exponential', 'hyperbolic', or 'harmonic'.

    Returns:
        JSON string with fitted parameters, R-squared, and predicted rates.
    """
    if model not in _MODELS:
        raise ValueError(f"Unknown model: {model}. Must be one of: {list(_MODELS.keys())}")

    if not production_data:
        raise ValueError("production_data is empty -- provide at least 3 data points")

    # Extract time and rate arrays
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

    # Remove zero/negative rates
    valid = q > 0
    t = t[valid]
    q = q[valid]
    if len(t) < 3:
        raise ValueError("Need at least 3 non-zero data points for curve fitting")

    func, param_names, p0_template, bounds = _MODELS[model]

    # Scale initial guess to data
    p0 = list(p0_template)
    p0[0] = float(q[0])  # qi
    if model == "duong":
        # For Duong, p0 = [qi, a, m] -- a and m use template defaults
        pass
    else:
        p0[1] = max(0.001, float((q[0] - q[-1]) / (q[0] * (t[-1] - t[0] + 1))))  # Di estimate

    try:
        popt, pcov = curve_fit(func, t, q, p0=p0, bounds=bounds, maxfev=10000)
    except RuntimeError as e:
        raise ValueError(f"Curve fitting failed to converge: {e}") from e

    q_pred = func(t, *popt)

    # R-squared
    ss_res = np.sum((q - q_pred) ** 2)
    ss_tot = np.sum((q - np.mean(q)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    params = dict(zip(param_names, [round(float(p), 6) for p in popt]))

    # Standard errors from covariance
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

    from petro_mcp._pro import is_pro
    if not is_pro():
        result["pro_hint"] = (
            "To fit decline curves for multiple wells at once with "
            "P10/P50/P90 type curves, see PetroSuite Pro at petropt.com/pro"
        )
        result["workspace_hint"] = "Save this analysis to your Petropt workspace: https://tools.petropt.com"

    return json.dumps(result, indent=2)


def calculate_eur(
    qi: float,
    Di: float = 0.0,
    b: float = 0.0,
    economic_limit: float = 5.0,
    model: str = "hyperbolic",
    max_time: float = 600,
    Dmin: float = 0.005,
    a: float = 1.0,
    m: float = 1.1,
) -> str:
    """Calculate Estimated Ultimate Recovery using decline parameters.

    Args:
        qi: Initial production rate (bbl/day or Mcf/day).
        Di: Initial decline rate (1/month, nominal). Used by exponential,
            hyperbolic, harmonic, and modified_hyperbolic models.
        b: Arps b-factor (0 = exponential, 1 = harmonic, 0-2 = hyperbolic).
        economic_limit: Minimum economic rate (same units as qi).
        model: Decline model - 'exponential', 'hyperbolic', 'harmonic',
            'modified_hyperbolic', or 'duong'.
        max_time: Maximum time in months to integrate.
        Dmin: Minimum terminal decline rate (1/month) for modified_hyperbolic.
        a: Duong intercept parameter (typically 0.5-2.0).
        m: Duong slope parameter (typically 1.0-1.5).

    Returns:
        JSON string with EUR, time to economic limit, and cumulative profile.
    """
    if qi <= 0:
        raise ValueError("qi must be positive")
    if model not in _MODELS:
        raise ValueError(f"Unknown model: {model}. Must be one of: {list(_MODELS.keys())}")

    func, param_names, _, _ = _MODELS[model]

    # Build the parameter dict from available arguments
    available_params = {"qi": qi, "Di": Di, "b": b, "Dmin": Dmin, "a": a, "m": m}

    # Validate Di for models that need it
    if "Di" in param_names and Di <= 0:
        raise ValueError("Di must be positive")

    # Clip b-factor for models that use it
    if "b" in param_names:
        b = float(np.clip(b, 0.001, 2.0))
        available_params["b"] = b

    # Assemble ordered parameter list for the model function
    model_params = [available_params[p] for p in param_names]

    # Build time array and compute rates
    t = np.arange(0, max_time + 1, dtype=float)
    rates = func(t, *model_params)

    # Enforce non-negative
    rates = np.maximum(rates, 0.0)

    # Find time to economic limit
    below_limit = np.where(rates < economic_limit)[0]
    if len(below_limit) > 0:
        econ_time = int(below_limit[0])
        rates = rates[:econ_time + 1]
        t = t[:econ_time + 1]
    else:
        econ_time = int(max_time)

    # EUR via trapezoidal integration (rate is per day, time is months)
    # Convert: rate (bbl/day) * 30.44 (days/month) = bbl/month
    days_per_month = 30.44
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    eur_bbl = float(_trapz(rates, t)) * days_per_month

    # Cumulative production at select intervals
    milestones = {}
    for yr in [1, 3, 5, 10, 20]:
        mo = yr * 12
        if mo <= len(rates):
            cum = float(_trapz(rates[:mo + 1], t[:mo + 1])) * days_per_month
            milestones[f"cum_{yr}yr"] = round(cum, 0)

    # Build output parameters dict (only include params relevant to the model)
    out_params = {p: available_params[p] for p in param_names}

    result = {
        "model": model,
        "parameters": out_params,
        "economic_limit": economic_limit,
        "eur": round(eur_bbl, 0),
        "eur_unit": "bbl (or Mcf if gas)",
        "time_to_economic_limit_months": econ_time,
        "time_to_economic_limit_years": round(econ_time / 12, 1),
        "cumulative_milestones": milestones,
        "final_rate": round(float(rates[-1]), 2) if len(rates) > 0 else 0,
    }
    return json.dumps(result, indent=2)
