"""Rate Transient Analysis (RTA) tools for the petro-mcp server.

Provides production-based reservoir characterization without shut-in tests.
Implements Blasingame, Agarwal-Gardner, NPI, Flowing Material Balance,
and sqrt(t) linear flow analysis methods.

References:
- Blasingame, T.A. et al. (1991). "Decline Curve Analysis for Variable
  Pressure Drop/Variable Flowrate Systems." SPE-21513.
- Agarwal, R.G., Gardner, D.C. et al. (1999). "Analyzing Well Production
  Data Using Combined Type Curve and Decline Curve Concepts." SPE-57916.
- Mattar, L. & Anderson, D. (2005). "Dynamic Material Balance." JCPT.
"""

from __future__ import annotations

import json
import math

import numpy as np


def calculate_normalized_rate(
    rate: list[float],
    flowing_pressure: list[float],
    initial_pressure: float,
) -> str:
    """Normalize rate by pressure drawdown: q / (Pi - Pwf).

    This removes the effect of variable flowing pressure from production
    data, making it suitable for type curve analysis.

    Args:
        rate: Production rates (bbl/d or Mcf/d).
        flowing_pressure: Bottomhole flowing pressures (psi).
        initial_pressure: Initial reservoir pressure (psi).

    Returns:
        JSON string with normalized rates and diagnostics.
    """
    if initial_pressure <= 0:
        raise ValueError("initial_pressure must be positive")
    if len(rate) != len(flowing_pressure):
        raise ValueError("rate and flowing_pressure must have the same length")
    if len(rate) == 0:
        raise ValueError("rate must not be empty")

    q = np.asarray(rate, dtype=float)
    pwf = np.asarray(flowing_pressure, dtype=float)
    dp = initial_pressure - pwf

    normalized = np.where(dp > 0, q / dp, 0.0)

    result = {
        "normalized_rate": [round(float(v), 6) for v in normalized],
        "drawdown": [round(float(v), 2) for v in dp],
        "num_points": len(rate),
        "avg_normalized_rate": round(float(np.mean(normalized[dp > 0])) if np.any(dp > 0) else 0.0, 6),
        "units": "rate_unit / psi",
    }
    return json.dumps(result, indent=2)


def calculate_material_balance_time(
    cumulative_production: list[float],
    rate: list[float],
) -> str:
    """Compute material balance time: tMB = Np / q.

    The Blasingame x-axis transform that converts variable-rate production
    into an equivalent constant-rate time.

    Args:
        cumulative_production: Cumulative production values.
        rate: Instantaneous production rates (same units as cumulative per time).

    Returns:
        JSON string with material balance times.
    """
    if len(cumulative_production) != len(rate):
        raise ValueError("cumulative_production and rate must have the same length")
    if len(rate) == 0:
        raise ValueError("rate must not be empty")

    np_arr = np.asarray(cumulative_production, dtype=float)
    q = np.asarray(rate, dtype=float)

    tmb = np.where(q > 0, np_arr / q, 0.0)

    result = {
        "material_balance_time": [round(float(v), 4) for v in tmb],
        "num_points": len(rate),
        "max_mbt": round(float(np.max(tmb)), 4),
        "units": "time (same as input rate time basis)",
    }
    return json.dumps(result, indent=2)


def calculate_blasingame_variables(
    times: list[float],
    rates: list[float],
    cumulative: list[float],
    flowing_pressures: list[float],
    initial_pressure: float,
) -> str:
    """Compute Blasingame rate-normalized integral and derivative.

    Blasingame type curve variables:
    - Rate-normalized cumulative: (Pi - Pwf) * Np / q  (integral)
    - Rate-integral: (1/tMB) * integral(q/dp, dtMB)
    - Rate-integral-derivative: -d(rate_integral)/d(ln(tMB))

    Args:
        times: Time values (days or months).
        rates: Production rates.
        cumulative: Cumulative production.
        flowing_pressures: Bottomhole flowing pressures (psi).
        initial_pressure: Initial reservoir pressure (psi).

    Returns:
        JSON string with Blasingame variables for type curve matching.
    """
    n = len(times)
    if not (n == len(rates) == len(cumulative) == len(flowing_pressures)):
        raise ValueError("All input arrays must have the same length")
    if n < 3:
        raise ValueError("Need at least 3 data points")
    if initial_pressure <= 0:
        raise ValueError("initial_pressure must be positive")

    t = np.asarray(times, dtype=float)
    q = np.asarray(rates, dtype=float)
    np_cum = np.asarray(cumulative, dtype=float)
    pwf = np.asarray(flowing_pressures, dtype=float)
    dp = initial_pressure - pwf

    # Normalized rate: q / dp
    qn = np.where(dp > 0, q / dp, 0.0)

    # Material balance time
    tmb = np.where(q > 0, np_cum / q, 0.0)

    # Rate-normalized integral: (1/tMB) * integral(qn, dtMB)
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    rate_integral = np.zeros(n)
    for i in range(1, n):
        if tmb[i] > 0:
            rate_integral[i] = _trapz(qn[:i + 1], tmb[:i + 1]) / tmb[i]
        else:
            rate_integral[i] = qn[i]

    # Rate-integral-derivative: -d(rate_integral)/d(ln(tMB))
    rate_integral_deriv = np.zeros(n)
    for i in range(1, n - 1):
        if tmb[i + 1] > tmb[i - 1] and tmb[i] > 0:
            d_ln_tmb = math.log(tmb[i + 1]) - math.log(tmb[i - 1]) if tmb[i - 1] > 0 and tmb[i + 1] > 0 else 1.0
            if d_ln_tmb != 0:
                rate_integral_deriv[i] = -(rate_integral[i + 1] - rate_integral[i - 1]) / d_ln_tmb
    # Forward/backward difference at boundaries
    if n > 1 and tmb[1] > 0 and tmb[0] >= 0:
        if tmb[1] > 0 and tmb[0] > 0:
            d_ln = math.log(tmb[1]) - math.log(tmb[0])
            if d_ln != 0:
                rate_integral_deriv[0] = -(rate_integral[1] - rate_integral[0]) / d_ln
    if n > 1 and tmb[-1] > 0 and tmb[-2] > 0:
        d_ln = math.log(tmb[-1]) - math.log(tmb[-2])
        if d_ln != 0:
            rate_integral_deriv[-1] = -(rate_integral[-1] - rate_integral[-2]) / d_ln

    result = {
        "material_balance_time": [round(float(v), 4) for v in tmb],
        "normalized_rate": [round(float(v), 6) for v in qn],
        "rate_integral": [round(float(v), 6) for v in rate_integral],
        "rate_integral_derivative": [round(float(v), 6) for v in rate_integral_deriv],
        "num_points": n,
        "method": "Blasingame",
    }
    return json.dumps(result, indent=2)


def calculate_agarwal_gardner_variables(
    times: list[float],
    rates: list[float],
    cumulative: list[float],
    flowing_pressures: list[float],
    initial_pressure: float,
) -> str:
    """Compute Agarwal-Gardner rate-normalized variables.

    AG type curve analysis uses:
    - x-axis: material balance time (tMB = Np/q)
    - y-axis: q / (Pi - Pwf)  (rate-normalized)
    - Also computes the AG inverse (Pi - Pwf) / q for derivative analysis

    Args:
        times: Time values (days or months).
        rates: Production rates.
        cumulative: Cumulative production.
        flowing_pressures: Bottomhole flowing pressures (psi).
        initial_pressure: Initial reservoir pressure (psi).

    Returns:
        JSON string with Agarwal-Gardner variables.
    """
    n = len(times)
    if not (n == len(rates) == len(cumulative) == len(flowing_pressures)):
        raise ValueError("All input arrays must have the same length")
    if n < 3:
        raise ValueError("Need at least 3 data points")
    if initial_pressure <= 0:
        raise ValueError("initial_pressure must be positive")

    q = np.asarray(rates, dtype=float)
    np_cum = np.asarray(cumulative, dtype=float)
    pwf = np.asarray(flowing_pressures, dtype=float)
    dp = initial_pressure - pwf

    # Material balance time
    tmb = np.where(q > 0, np_cum / q, 0.0)

    # Rate-normalized: q / dp
    qn = np.where(dp > 0, q / dp, 0.0)

    # Inverse: dp / q (for derivative analysis)
    inv_qn = np.where(q > 0, dp / q, 0.0)

    # AG cumulative-normalized: Np / dp (contacted volume indicator)
    cum_normalized = np.where(dp > 0, np_cum / dp, 0.0)

    result = {
        "material_balance_time": [round(float(v), 4) for v in tmb],
        "normalized_rate": [round(float(v), 6) for v in qn],
        "inverse_normalized_rate": [round(float(v), 4) for v in inv_qn],
        "cumulative_normalized": [round(float(v), 4) for v in cum_normalized],
        "num_points": n,
        "method": "Agarwal-Gardner",
    }
    return json.dumps(result, indent=2)


def calculate_npi_variables(
    times: list[float],
    rates: list[float],
    flowing_pressures: list[float],
    initial_pressure: float,
) -> str:
    """Compute Normalized Pressure Integral (NPI) variables.

    NPI analysis integrates the pressure-normalized rate over time to
    smooth noisy production data for flowing material balance.

    NPI = (1/t) * integral( (Pi - Pwf(tau)) / q(tau), dtau )

    Args:
        times: Time values (days or months).
        rates: Production rates.
        flowing_pressures: Bottomhole flowing pressures (psi).
        initial_pressure: Initial reservoir pressure (psi).

    Returns:
        JSON string with NPI variables.
    """
    n = len(times)
    if not (n == len(rates) == len(flowing_pressures)):
        raise ValueError("All input arrays must have the same length")
    if n < 3:
        raise ValueError("Need at least 3 data points")
    if initial_pressure <= 0:
        raise ValueError("initial_pressure must be positive")

    t = np.asarray(times, dtype=float)
    q = np.asarray(rates, dtype=float)
    pwf = np.asarray(flowing_pressures, dtype=float)
    dp = initial_pressure - pwf

    # dp / q (pressure-normalized inverse rate)
    inv_qn = np.where(q > 0, dp / q, 0.0)

    # Normalized Pressure Integral
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    npi = np.zeros(n)
    for i in range(1, n):
        if t[i] > t[0]:
            npi[i] = _trapz(inv_qn[:i + 1], t[:i + 1]) / (t[i] - t[0])

    # NPI derivative: d(NPI)/d(ln(t))
    npi_deriv = np.zeros(n)
    for i in range(1, n - 1):
        if t[i + 1] > 0 and t[i - 1] > 0 and t[i] > 0:
            d_ln_t = math.log(t[i + 1]) - math.log(t[i - 1])
            if d_ln_t != 0:
                npi_deriv[i] = (npi[i + 1] - npi[i - 1]) / d_ln_t

    result = {
        "times": [round(float(v), 4) for v in t],
        "inverse_normalized_rate": [round(float(v), 4) for v in inv_qn],
        "npi": [round(float(v), 4) for v in npi],
        "npi_derivative": [round(float(v), 4) for v in npi_deriv],
        "num_points": n,
        "method": "NPI (Normalized Pressure Integral)",
    }
    return json.dumps(result, indent=2)


def calculate_flowing_material_balance(
    rates: list[float],
    flowing_pressures: list[float],
    initial_pressure: float,
    fluid_fvf: float,
    total_compressibility: float,
) -> str:
    """Flowing Material Balance (FMB) analysis.

    Plots q/(Pi-Pwf) vs cumulative normalized pressure (Np*Bf*ct) to
    estimate original oil/gas in place from the slope.

    FMB equation: q/(Pi-Pwf) = -slope * N_p + intercept
    where OOIP = intercept / slope (contacted pore volume)

    Args:
        rates: Production rates (bbl/d or Mcf/d).
        flowing_pressures: Bottomhole flowing pressures (psi).
        initial_pressure: Initial reservoir pressure (psi).
        fluid_fvf: Formation volume factor (rb/stb for oil, rcf/scf for gas).
        total_compressibility: Total system compressibility (1/psi).

    Returns:
        JSON string with FMB plot data and estimated OOIP/OGIP.
    """
    if initial_pressure <= 0:
        raise ValueError("initial_pressure must be positive")
    if fluid_fvf <= 0:
        raise ValueError("fluid_fvf must be positive")
    if total_compressibility <= 0:
        raise ValueError("total_compressibility must be positive")
    if len(rates) != len(flowing_pressures):
        raise ValueError("rates and flowing_pressures must have the same length")
    if len(rates) < 3:
        raise ValueError("Need at least 3 data points")

    q = np.asarray(rates, dtype=float)
    pwf = np.asarray(flowing_pressures, dtype=float)
    dp = initial_pressure - pwf

    # Normalized rate: q / dp
    qn = np.where(dp > 0, q / dp, 0.0)

    # Cumulative production (trapezoidal, assume unit time steps)
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    cum = np.zeros(len(q))
    for i in range(1, len(q)):
        cum[i] = cum[i - 1] + (q[i] + q[i - 1]) / 2.0

    # Normalized cumulative: Np * Bf * ct
    cum_normalized = cum * fluid_fvf * total_compressibility

    # Linear regression on valid points (where dp > 0 and q > 0)
    valid = (dp > 0) & (q > 0)
    if np.sum(valid) < 2:
        raise ValueError("Need at least 2 valid data points (positive rate and drawdown)")

    x = cum_normalized[valid]
    y = qn[valid]

    # Least-squares fit: y = slope * x + intercept
    coeffs = np.polyfit(x, y, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    # OOIP estimate: x-intercept of the line = -intercept/slope
    ooip_estimate = None
    if slope < 0 and intercept > 0:
        # x-intercept = -intercept / slope  =>  Np at abandonment
        # OOIP = x_intercept / (Bf * ct)
        x_intercept = -intercept / slope
        ooip_estimate = round(x_intercept / (fluid_fvf * total_compressibility), 0)

    # R-squared
    y_pred = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    result = {
        "normalized_rate": [round(float(v), 6) for v in qn],
        "cumulative_normalized": [round(float(v), 6) for v in cum_normalized],
        "fmb_slope": round(slope, 6),
        "fmb_intercept": round(intercept, 6),
        "r_squared": round(r_squared, 6),
        "ooip_estimate": ooip_estimate,
        "ooip_unit": "stb (or scf if gas)",
        "num_valid_points": int(np.sum(valid)),
        "method": "Flowing Material Balance",
    }
    return json.dumps(result, indent=2)


def calculate_sqrt_time_analysis(
    rates: list[float],
    times: list[float],
    flowing_pressures: list[float],
    initial_pressure: float,
) -> str:
    """Square root of time analysis for linear flow identification.

    During linear flow (fracture-dominated), 1/q vs sqrt(t) is a straight
    line. The slope gives sqrt(k)*xf (flow capacity * fracture half-length).

    Args:
        rates: Production rates (bbl/d or Mcf/d).
        times: Time values (days).
        flowing_pressures: Bottomhole flowing pressures (psi).
        initial_pressure: Initial reservoir pressure (psi).

    Returns:
        JSON string with sqrt(t) analysis results and linear flow slope.
    """
    n = len(rates)
    if not (n == len(times) == len(flowing_pressures)):
        raise ValueError("All input arrays must have the same length")
    if n < 3:
        raise ValueError("Need at least 3 data points")
    if initial_pressure <= 0:
        raise ValueError("initial_pressure must be positive")

    q = np.asarray(rates, dtype=float)
    t = np.asarray(times, dtype=float)
    pwf = np.asarray(flowing_pressures, dtype=float)
    dp = initial_pressure - pwf

    sqrt_t = np.sqrt(np.maximum(t, 0.0))

    # Pressure-normalized inverse rate: (Pi - Pwf) / q
    inv_qn = np.where(q > 0, dp / q, 0.0)

    # Linear regression on valid points
    valid = (q > 0) & (t > 0) & (dp > 0)
    if np.sum(valid) < 2:
        raise ValueError("Need at least 2 valid data points")

    x = sqrt_t[valid]
    y = inv_qn[valid]

    coeffs = np.polyfit(x, y, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    # R-squared
    y_pred = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # End of linear flow detection: find where data deviates from line
    residuals = np.abs(y - y_pred)
    threshold = 2.0 * np.std(residuals) if len(residuals) > 0 else 0
    end_linear = None
    for i in range(len(residuals)):
        if residuals[i] > threshold and i > len(residuals) * 0.3:
            end_linear = round(float(t[valid][i]), 2)
            break

    result = {
        "sqrt_time": [round(float(v), 4) for v in sqrt_t],
        "inverse_normalized_rate": [round(float(v), 4) for v in inv_qn],
        "linear_flow_slope": round(slope, 6),
        "linear_flow_intercept": round(intercept, 6),
        "r_squared": round(r_squared, 6),
        "end_of_linear_flow_time": end_linear,
        "num_valid_points": int(np.sum(valid)),
        "method": "Square Root of Time Analysis",
    }
    return json.dumps(result, indent=2)


def calculate_rta_permeability(
    slope_from_linear_flow: float,
    net_pay_ft: float,
    porosity: float,
    viscosity_cp: float,
    total_compressibility: float,
    fracture_half_length_ft: float | None = None,
) -> str:
    """Extract permeability from RTA linear flow slope.

    For oil, the linear flow slope m relates to reservoir properties via:

        m = 4.064 * B * mu / (h * xf * sqrt(k * phi * mu * ct))

    where m is the slope from sqrt(t) analysis, B is FVF, mu is viscosity,
    h is net pay, xf is fracture half-length, k is permeability, phi is
    porosity, and ct is total compressibility.

    If fracture_half_length is not provided, returns sqrt(k)*xf instead.

    Args:
        slope_from_linear_flow: Slope from sqrt(t) analysis (psi*d/bbl/d^0.5).
        net_pay_ft: Net pay thickness (ft).
        porosity: Porosity (fraction, 0-1).
        viscosity_cp: Fluid viscosity (cp).
        total_compressibility: Total compressibility (1/psi).
        fracture_half_length_ft: Fracture half-length (ft). If None, returns
            sqrt(k)*xf product.

    Returns:
        JSON string with permeability estimate or sqrt(k)*xf.
    """
    if slope_from_linear_flow <= 0:
        raise ValueError("slope_from_linear_flow must be positive")
    if net_pay_ft <= 0:
        raise ValueError("net_pay_ft must be positive")
    if not (0 < porosity <= 1):
        raise ValueError("porosity must be between 0 and 1 (exclusive of 0)")
    if viscosity_cp <= 0:
        raise ValueError("viscosity_cp must be positive")
    if total_compressibility <= 0:
        raise ValueError("total_compressibility must be positive")

    # Linear flow constant for oil (field units): 4.064
    # m = C * B_o * mu / (h * xf * sqrt(k * phi * mu * ct))
    # Rearranging: sqrt(k) * xf = C * mu / (m * h * sqrt(phi * mu * ct))
    # We use B_o = 1 as a default (user can adjust slope accordingly)
    C = 4.064
    sqrt_phi_mu_ct = math.sqrt(porosity * viscosity_cp * total_compressibility)

    if sqrt_phi_mu_ct <= 0:
        raise ValueError("Product of porosity * viscosity * compressibility must be positive")

    sqrt_k_xf = C * viscosity_cp / (slope_from_linear_flow * net_pay_ft * sqrt_phi_mu_ct)

    result: dict = {
        "sqrt_k_times_xf": round(sqrt_k_xf, 6),
        "sqrt_k_times_xf_unit": "md^0.5 * ft",
        "net_pay_ft": net_pay_ft,
        "porosity": porosity,
        "viscosity_cp": viscosity_cp,
        "total_compressibility": total_compressibility,
    }

    if fracture_half_length_ft is not None:
        if fracture_half_length_ft <= 0:
            raise ValueError("fracture_half_length_ft must be positive")
        sqrt_k = sqrt_k_xf / fracture_half_length_ft
        k = sqrt_k ** 2
        result["fracture_half_length_ft"] = fracture_half_length_ft
        result["permeability_md"] = round(k, 6)
        result["sqrt_permeability"] = round(sqrt_k, 6)
    else:
        result["note"] = (
            "Provide fracture_half_length_ft to solve for permeability. "
            "Without it, only the sqrt(k)*xf product can be determined."
        )

    result["method"] = "RTA Permeability from Linear Flow"
    return json.dumps(result, indent=2)
