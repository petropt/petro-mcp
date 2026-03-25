"""Advanced decline curve analysis tools using petbox-dca.

Wraps petbox-dca models for unconventional/shale well analysis:
- PLE (Power Law Exponential)
- Duong
- SEPD (Stretched Exponential Production Decline)
- THM (Transient Hyperbolic Model)

All models use scipy.optimize.curve_fit for parameter estimation,
with petbox-dca providing the forward model computations.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

import petbox.dca as dca

DAYS_PER_MONTH = 30.44


def _extract_time_rate(
    production_data: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract time (days) and rate arrays from production data.

    Accepts dicts with 'time' (months) and 'rate' keys, or 'oil'/'gas' keys
    with implicit monthly spacing.  Returned time is in days (petbox-dca units).
    """
    if not production_data:
        raise ValueError("production_data is empty -- provide at least 3 data points")

    if "time" in production_data[0]:
        t_months = np.array([d["time"] for d in production_data], dtype=float)
        q = np.array([d["rate"] for d in production_data], dtype=float)
    else:
        t_months = np.arange(len(production_data), dtype=float)
        for key in ("oil", "gas", "rate"):
            if key in production_data[0]:
                q = np.array([d[key] for d in production_data], dtype=float)
                break
        else:
            raise ValueError("Production data must have 'rate', 'oil', or 'gas' key")

    # Remove zero/negative rates
    valid = q > 0
    t_months = t_months[valid]
    q = q[valid]
    if len(t_months) < 3:
        raise ValueError("Need at least 3 non-zero data points for curve fitting")

    # Convert months to days for petbox-dca
    t_days = t_months * DAYS_PER_MONTH
    return t_days, q


def _r_squared(q_obs: np.ndarray, q_pred: np.ndarray) -> float:
    """Compute R-squared goodness of fit."""
    ss_res = np.sum((q_obs - q_pred) ** 2)
    ss_tot = np.sum((q_obs - np.mean(q_obs)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _residuals_summary(q_obs: np.ndarray, q_pred: np.ndarray) -> dict[str, float]:
    """Compute residual statistics."""
    resid = q_obs - q_pred
    return {
        "mean": round(float(np.mean(resid)), 2),
        "std": round(float(np.std(resid)), 2),
        "max_abs": round(float(np.max(np.abs(resid))), 2),
    }


# ---------------------------------------------------------------------------
# PLE (Power Law Exponential) model
# ---------------------------------------------------------------------------

def _ple_rate(t: np.ndarray, qi: float, Di: float, Dinf: float, n: float) -> np.ndarray:
    """PLE forward model for curve_fit.  t in days."""
    # Clamp parameters to valid ranges
    Dinf = max(Dinf, 0.0)
    n = np.clip(n, 0.01, 0.99)
    model = dca.PLE(qi=qi, Di=Di, Dinf=Dinf, n=n)
    return model.rate(t)


def fit_ple_decline(production_data: list[dict[str, float]]) -> str:
    """Fit Power Law Exponential decline model to production data.

    The PLE model (Ilk et al., 2008) captures transient and boundary-dominated
    flow regimes common in tight/shale reservoirs.

    q(t) = qi * exp(-Di * t^n - Dinf * t)

    Args:
        production_data: List of dicts with 'time' (months from first production)
            and 'rate' (bbl/day or Mcf/day) keys.  Alternatively, 'oil'/'gas' keys
            with implicit monthly spacing.

    Returns:
        JSON string with model parameters, R-squared, and predicted rates.
    """
    t_days, q = _extract_time_rate(production_data)

    # Ensure t starts at 1 day minimum (PLE is undefined at t=0)
    t_days = np.maximum(t_days, 1.0)

    qi0 = float(q[0])
    p0 = [qi0, 0.001, 0.0001, 0.5]
    bounds = ([0, 1e-6, 0, 0.01], [qi0 * 5, 1.0, 0.01, 0.99])

    try:
        popt, pcov = curve_fit(_ple_rate, t_days, q, p0=p0, bounds=bounds, maxfev=20000)
    except RuntimeError as e:
        raise ValueError(f"PLE curve fitting failed to converge: {e}") from e

    q_pred = _ple_rate(t_days, *popt)
    perr = np.sqrt(np.diag(pcov))

    param_names = ["qi", "Di", "Dinf", "n"]
    params = dict(zip(param_names, [round(float(p), 6) for p in popt]))
    param_errors = dict(zip(param_names, [round(float(e), 6) for e in perr]))

    result: dict[str, Any] = {
        "model": "ple",
        "parameters": params,
        "parameter_errors": param_errors,
        "r_squared": round(_r_squared(q, q_pred), 6),
        "num_data_points": len(t_days),
        "predicted_rates": [round(float(v), 2) for v in q_pred],
        "residuals_summary": _residuals_summary(q, q_pred),
        "units": {
            "qi": "vol/day",
            "Di": "1/day",
            "Dinf": "1/day",
            "n": "dimensionless",
            "time_input": "months (converted to days internally)",
        },
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Duong model
# ---------------------------------------------------------------------------

def _duong_rate(t: np.ndarray, qi: float, a: float, m: float) -> np.ndarray:
    """Duong forward model for curve_fit.  t in days."""
    a = max(a, 1.0)
    m = max(m, 1.001)
    model = dca.Duong(qi=qi, a=a, m=m)
    return model.rate(t)


def fit_duong_decline(production_data: list[dict[str, float]]) -> str:
    """Fit Duong decline model to production data.

    The Duong model (2011) is designed for fracture-dominated flow in
    unconventional/shale reservoirs.  Widely used for tight oil and shale gas.

    Args:
        production_data: List of dicts with 'time' (months from first production)
            and 'rate' (bbl/day or Mcf/day) keys.  Alternatively, 'oil'/'gas' keys
            with implicit monthly spacing.

    Returns:
        JSON string with model parameters, R-squared, and predicted rates.
    """
    t_days, q = _extract_time_rate(production_data)

    # Duong model requires t >= 1 day
    t_days = np.maximum(t_days, 1.0)

    qi0 = float(q[0])
    p0 = [qi0, 1.5, 1.1]
    bounds = ([0, 1.0, 1.001], [qi0 * 5, 10.0, 3.0])

    try:
        popt, pcov = curve_fit(_duong_rate, t_days, q, p0=p0, bounds=bounds, maxfev=20000)
    except RuntimeError as e:
        raise ValueError(f"Duong curve fitting failed to converge: {e}") from e

    q_pred = _duong_rate(t_days, *popt)
    perr = np.sqrt(np.diag(pcov))

    param_names = ["qi", "a", "m"]
    params = dict(zip(param_names, [round(float(p), 6) for p in popt]))
    param_errors = dict(zip(param_names, [round(float(e), 6) for e in perr]))

    result: dict[str, Any] = {
        "model": "duong",
        "parameters": params,
        "parameter_errors": param_errors,
        "r_squared": round(_r_squared(q, q_pred), 6),
        "num_data_points": len(t_days),
        "predicted_rates": [round(float(v), 2) for v in q_pred],
        "residuals_summary": _residuals_summary(q, q_pred),
        "units": {
            "qi": "vol/day (defined at t=1 day)",
            "a": "dimensionless",
            "m": "dimensionless",
            "time_input": "months (converted to days internally)",
        },
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# SEPD (Stretched Exponential) model
# ---------------------------------------------------------------------------

def _sepd_rate(t: np.ndarray, qi: float, tau: float, n: float) -> np.ndarray:
    """SEPD forward model for curve_fit.  t in days."""
    tau = max(tau, 1e-6)
    n = np.clip(n, 1e-6, 0.999)
    model = dca.SE(qi=qi, tau=tau, n=n)
    return model.rate(t)


def fit_sepd_decline(production_data: list[dict[str, float]]) -> str:
    """Fit Stretched Exponential Production Decline model to production data.

    The SEPD model (Valko, 2009) uses a stretched exponential function to
    characterize production decline.  Effective for unconventional reservoirs
    with heterogeneous fracture networks.

    q(t) = qi * exp(-(t/tau)^n)

    Args:
        production_data: List of dicts with 'time' (months from first production)
            and 'rate' (bbl/day or Mcf/day) keys.  Alternatively, 'oil'/'gas' keys
            with implicit monthly spacing.

    Returns:
        JSON string with model parameters, R-squared, and predicted rates.
    """
    t_days, q = _extract_time_rate(production_data)

    # SEPD needs t > 0
    t_days = np.maximum(t_days, 1.0)

    qi0 = float(q[0])
    p0 = [qi0, 500.0, 0.5]
    bounds = ([0, 1e-3, 1e-6], [qi0 * 5, 10000.0, 0.999])

    try:
        popt, pcov = curve_fit(_sepd_rate, t_days, q, p0=p0, bounds=bounds, maxfev=20000)
    except RuntimeError as e:
        raise ValueError(f"SEPD curve fitting failed to converge: {e}") from e

    q_pred = _sepd_rate(t_days, *popt)
    perr = np.sqrt(np.diag(pcov))

    param_names = ["qi", "tau", "n"]
    params = dict(zip(param_names, [round(float(p), 6) for p in popt]))
    param_errors = dict(zip(param_names, [round(float(e), 6) for e in perr]))

    result: dict[str, Any] = {
        "model": "sepd",
        "parameters": params,
        "parameter_errors": param_errors,
        "r_squared": round(_r_squared(q, q_pred), 6),
        "num_data_points": len(t_days),
        "predicted_rates": [round(float(v), 2) for v in q_pred],
        "residuals_summary": _residuals_summary(q, q_pred),
        "units": {
            "qi": "vol/day",
            "tau": "day^n",
            "n": "dimensionless (0 < n < 1)",
            "time_input": "months (converted to days internally)",
        },
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Forecast using any advanced model
# ---------------------------------------------------------------------------

_MODEL_CLASSES = {
    "ple": (dca.PLE, ["qi", "Di", "Dinf", "n"]),
    "duong": (dca.Duong, ["qi", "a", "m"]),
    "sepd": (dca.SE, ["qi", "tau", "n"]),
    "thm": (dca.THM, ["qi", "Di", "bi", "bf", "telf"]),
}


def forecast_advanced_decline(
    model: str,
    parameters: dict[str, float],
    forecast_months: int = 360,
    economic_limit: float = 5.0,
) -> str:
    """Forecast production using an advanced decline model from petbox-dca.

    Generates a rate-time forecast and cumulative production profile for
    PLE, Duong, SEPD, or THM (Transient Hyperbolic) models.

    Args:
        model: Model name - 'ple', 'duong', 'sepd', or 'thm'.
        parameters: Dict of model parameters (from fitting results).
            PLE: qi, Di, Dinf, n
            Duong: qi, a, m
            SEPD: qi, tau, n
            THM: qi, Di, bi, bf, telf (optional: bterm, tterm)
        forecast_months: Number of months to forecast (default 360 = 30 years).
        economic_limit: Minimum economic rate in vol/day (default 5.0).

    Returns:
        JSON string with forecast rates, cumulative production, and EUR.
    """
    if model not in _MODEL_CLASSES:
        raise ValueError(
            f"Unknown model: {model}. Must be one of: {list(_MODEL_CLASSES.keys())}"
        )

    cls, required_params = _MODEL_CLASSES[model]

    # Validate required parameters
    missing = [p for p in required_params if p not in parameters]
    if missing:
        raise ValueError(f"Missing required parameters for {model}: {missing}")

    # Build model instance
    if model == "thm":
        # THM has optional bterm/tterm
        kwargs = {p: parameters[p] for p in required_params}
        if "bterm" in parameters:
            kwargs["bterm"] = parameters["bterm"]
        if "tterm" in parameters:
            kwargs["tterm"] = parameters["tterm"]
        mdl = cls(**kwargs)
    else:
        kwargs = {p: parameters[p] for p in required_params}
        mdl = cls(**kwargs)

    # Generate time array in days
    t_days = np.arange(1, forecast_months * DAYS_PER_MONTH + 1, DAYS_PER_MONTH)
    rates = mdl.rate(t_days)
    rates = np.maximum(rates, 0.0)

    # Find economic limit
    below_limit = np.where(rates < economic_limit)[0]
    if len(below_limit) > 0:
        econ_idx = int(below_limit[0])
        rates = rates[: econ_idx + 1]
        t_days = t_days[: econ_idx + 1]
    econ_time_months = len(rates)

    # Cumulative via trapezoidal integration (rate is vol/day, t is days)
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    eur = float(_trapz(rates, t_days))

    # Milestones
    milestones: dict[str, float] = {}
    for yr in [1, 3, 5, 10, 20, 30]:
        mo = yr * 12
        if mo <= len(rates):
            cum_at = float(_trapz(rates[:mo], t_days[:mo]))
            milestones[f"cum_{yr}yr"] = round(cum_at, 0)

    # Monthly rate profile (sampled)
    monthly_rates = [round(float(r), 2) for r in rates]

    result: dict[str, Any] = {
        "model": model,
        "parameters": {k: round(float(v), 6) for k, v in parameters.items()},
        "forecast_months": forecast_months,
        "economic_limit": economic_limit,
        "eur": round(eur, 0),
        "eur_unit": "bbl (or Mcf if gas)",
        "time_to_economic_limit_months": econ_time_months,
        "time_to_economic_limit_years": round(econ_time_months / 12, 1),
        "cumulative_milestones": milestones,
        "final_rate": round(float(rates[-1]), 2) if len(rates) > 0 else 0,
        "monthly_rates": monthly_rates,
        "units": {
            "rates": "vol/day",
            "cumulative": "vol (bbl or Mcf)",
            "time": "months",
        },
    }
    return json.dumps(result, indent=2)
