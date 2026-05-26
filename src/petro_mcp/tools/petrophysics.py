"""Petrophysics log interpretation calculations.

Standard single-formula petrophysical calculations:
    - Vshale from gamma ray (linear, Larionov, Clavier)
    - Density porosity
    - Archie water saturation
    - Net pay with cutoffs
"""

from __future__ import annotations

import json
import math


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_positive(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value}")


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# 1. Vshale
# ---------------------------------------------------------------------------

_VSHALE_METHODS = {"linear", "larionov_tertiary", "larionov_older", "clavier"}


def calculate_vshale(
    gr: float,
    gr_clean: float,
    gr_shale: float,
    method: str = "linear",
) -> str:
    """Calculate shale volume from gamma ray log.

    Args:
        gr: Gamma ray reading (API units).
        gr_clean: GR in clean sand (API units).
        gr_shale: GR in pure shale (API units).
        method: One of linear, larionov_tertiary, larionov_older, clavier.

    Returns:
        JSON string with vshale (fraction v/v).
    """
    method = method.lower()
    if method not in _VSHALE_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Must be one of: {sorted(_VSHALE_METHODS)}"
        )
    _validate_finite("gr", gr)
    _validate_finite("gr_clean", gr_clean)
    _validate_finite("gr_shale", gr_shale)
    if gr_clean == gr_shale:
        raise ValueError("gr_clean and gr_shale must differ")

    igr = _clamp((gr - gr_clean) / (gr_shale - gr_clean))

    if method == "linear":
        vsh = igr
    elif method == "larionov_tertiary":
        vsh = 0.083 * (2 ** (3.7 * igr) - 1)
    elif method == "larionov_older":
        vsh = 0.33 * (2 ** (2.0 * igr) - 1)
    else:  # clavier
        vsh = 1.7 - math.sqrt(max(3.38 - (igr + 0.7) ** 2, 0.0))

    vsh = _clamp(vsh)

    return json.dumps({
        "vshale": round(vsh, 4),
        "igr": round(igr, 4),
        "method": method,
        "inputs": {"gr": gr, "gr_clean": gr_clean, "gr_shale": gr_shale},
        "units": {"vshale": "fraction v/v"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 2. Density porosity
# ---------------------------------------------------------------------------

def calculate_density_porosity(
    rhob: float,
    rho_matrix: float = 2.65,
    rho_fluid: float = 1.0,
) -> str:
    """Calculate porosity from bulk density log.

    Args:
        rhob: Bulk density (g/cc).
        rho_matrix: Matrix density (g/cc). Default 2.65 for sandstone.
        rho_fluid: Fluid density (g/cc). Default 1.0 for fresh water.

    Returns:
        JSON string with density_porosity (fraction v/v).
    """
    _validate_positive("rhob", rhob)
    _validate_positive("rho_matrix", rho_matrix)
    _validate_positive("rho_fluid", rho_fluid)
    if rho_matrix == rho_fluid:
        raise ValueError("rho_matrix and rho_fluid must differ")

    phi = _clamp((rho_matrix - rhob) / (rho_matrix - rho_fluid))

    return json.dumps({
        "density_porosity": round(phi, 4),
        "correlation": "Bulk density",
        "inputs": {
            "rhob": rhob,
            "rho_matrix": rho_matrix,
            "rho_fluid": rho_fluid,
        },
        "units": {"density_porosity": "fraction v/v", "rhob": "g/cc"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 3. Archie water saturation
# ---------------------------------------------------------------------------

def calculate_archie_sw(
    rt: float,
    phi: float,
    rw: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
) -> str:
    """Calculate water saturation using the Archie equation (clean sands).

    Sw = (a * Rw / (phi^m * Rt))^(1/n)

    Args:
        rt: True formation resistivity (ohm-m).
        phi: Porosity (fraction v/v, 0-1).
        rw: Formation water resistivity (ohm-m).
        a: Tortuosity factor. Default 1.0.
        m: Cementation exponent. Default 2.0.
        n: Saturation exponent. Default 2.0.

    Returns:
        JSON string with water_saturation (fraction v/v).
    """
    _validate_positive("rt", rt)
    _validate_positive("phi", phi)
    _validate_positive("rw", rw)
    _validate_positive("a", a)
    _validate_positive("m", m)
    _validate_positive("n", n)
    if phi > 1:
        raise ValueError(f"phi must be <= 1, got {phi}")

    sw_raw = (a * rw / (phi ** m * rt)) ** (1.0 / n)
    sw = _clamp(sw_raw)

    result = {
        "water_saturation": round(sw, 4),
        "hydrocarbon_saturation": round(1.0 - sw, 4),
        "correlation": "Archie (1942)",
        "inputs": {"rt": rt, "phi": phi, "rw": rw, "a": a, "m": m, "n": n},
        "units": {"water_saturation": "fraction v/v", "rt": "ohm-m"},
    }


    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# 4. Net pay
# ---------------------------------------------------------------------------

def calculate_net_pay(
    depths: list[float],
    phi: list[float],
    sw: list[float],
    vshale: list[float],
    phi_cutoff: float = 0.06,
    sw_cutoff: float = 0.5,
    vsh_cutoff: float = 0.5,
) -> str:
    """Determine net pay by applying porosity, Sw, and Vshale cutoffs.

    Args:
        depths: Measured depths (ft), monotonically increasing.
        phi: Porosity values (fraction v/v) at each depth.
        sw: Water saturation values (fraction v/v) at each depth.
        vshale: Shale volume values (fraction v/v) at each depth.
        phi_cutoff: Minimum porosity for pay. Default 0.06.
        sw_cutoff: Maximum water saturation for pay. Default 0.5.
        vsh_cutoff: Maximum Vshale for pay. Default 0.5.

    Returns:
        JSON string with net pay thickness, NTG, average properties,
        and per-sample pay flags.
    """
    n = len(depths)
    if n < 2:
        raise ValueError("At least 2 depth points required")
    if len(phi) != n or len(sw) != n or len(vshale) != n:
        raise ValueError(
            f"All arrays must have the same length. Got depths={n}, "
            f"phi={len(phi)}, sw={len(sw)}, vshale={len(vshale)}"
        )

    # Compute per-interval thickness (last sample gets same step as previous)
    thicknesses = []
    for i in range(n - 1):
        thicknesses.append(abs(depths[i + 1] - depths[i]))
    thicknesses.append(thicknesses[-1] if thicknesses else 0.0)

    gross = sum(thicknesses)

    # Apply cutoffs
    pay_flags = []
    net = 0.0
    phi_sum = 0.0
    sw_sum = 0.0
    vsh_sum = 0.0
    pay_thick_total = 0.0

    for i in range(n):
        is_pay = (
            phi[i] >= phi_cutoff
            and sw[i] <= sw_cutoff
            and vshale[i] <= vsh_cutoff
        )
        pay_flags.append(is_pay)
        if is_pay:
            h = thicknesses[i]
            net += h
            phi_sum += phi[i] * h
            sw_sum += sw[i] * h
            vsh_sum += vshale[i] * h
            pay_thick_total += h

    avg_phi = round(phi_sum / pay_thick_total, 4) if pay_thick_total > 0 else None
    avg_sw = round(sw_sum / pay_thick_total, 4) if pay_thick_total > 0 else None
    avg_vsh = round(vsh_sum / pay_thick_total, 4) if pay_thick_total > 0 else None
    ntg = round(net / gross, 4) if gross > 0 else 0.0

    result: dict = {
        "net_pay_ft": round(net, 2),
        "gross_thickness_ft": round(gross, 2),
        "net_to_gross": ntg,
        "avg_porosity_pay": avg_phi,
        "avg_sw_pay": avg_sw,
        "avg_vshale_pay": avg_vsh,
        "pay_flags": pay_flags,
        "num_pay_samples": sum(pay_flags),
        "cutoffs": {
            "phi_cutoff": phi_cutoff,
            "sw_cutoff": sw_cutoff,
            "vsh_cutoff": vsh_cutoff,
        },
        "units": {"thickness": "ft"},
    }
    if pay_thick_total == 0:
        result["note"] = "No intervals meet cutoff criteria"


    return json.dumps(result, indent=2)


