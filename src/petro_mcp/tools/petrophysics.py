"""Petrophysics log interpretation calculations.

Standard single-formula petrophysical calculations:
    - Vshale from gamma ray (linear, Larionov, Clavier)
    - Porosity (density, sonic, neutron-density, effective)
    - Water saturation (Archie, Simandoux, Indonesian)
    - Permeability (Timur, Coates)
    - Net pay with cutoffs
    - Hydrocarbon pore thickness (HPT)
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


def _validate_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _validate_fraction(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value}")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


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
# 3. Sonic porosity
# ---------------------------------------------------------------------------

_SONIC_METHODS = {"wyllie", "raymer"}


def calculate_sonic_porosity(
    dt: float,
    dt_matrix: float = 55.5,
    dt_fluid: float = 189.0,
    method: str = "wyllie",
) -> str:
    """Calculate porosity from sonic (compressional) log.

    Args:
        dt: Interval transit time (us/ft).
        dt_matrix: Matrix transit time (us/ft). Default 55.5 for sandstone.
        dt_fluid: Fluid transit time (us/ft). Default 189.0.
        method: 'wyllie' (time-average) or 'raymer' (Raymer-Hunt-Gardner).

    Returns:
        JSON string with sonic_porosity (fraction v/v).
    """
    method = method.lower()
    if method not in _SONIC_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Must be one of: {sorted(_SONIC_METHODS)}"
        )
    _validate_positive("dt", dt)
    _validate_positive("dt_matrix", dt_matrix)
    _validate_positive("dt_fluid", dt_fluid)
    if dt_matrix == dt_fluid:
        raise ValueError("dt_matrix and dt_fluid must differ")

    if method == "wyllie":
        phi = (dt - dt_matrix) / (dt_fluid - dt_matrix)
        corr = "Wyllie time-average"
    else:
        phi = 0.625 * (dt - dt_matrix) / dt
        corr = "Raymer-Hunt-Gardner"

    phi = _clamp(phi)

    return json.dumps({
        "sonic_porosity": round(phi, 4),
        "method": method,
        "correlation": corr,
        "inputs": {
            "dt": dt,
            "dt_matrix": dt_matrix,
            "dt_fluid": dt_fluid,
        },
        "units": {"sonic_porosity": "fraction v/v", "dt": "us/ft"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 4. Neutron-density porosity
# ---------------------------------------------------------------------------

def calculate_neutron_density_porosity(
    nphi: float,
    dphi: float,
) -> str:
    """Quick-look porosity from neutron-density combination.

    Args:
        nphi: Neutron porosity (fraction v/v).
        dphi: Density porosity (fraction v/v).

    Returns:
        JSON string with neutron_density_porosity (fraction v/v).
    """
    phi = math.sqrt((nphi ** 2 + dphi ** 2) / 2.0)
    phi = _clamp(phi)

    return json.dumps({
        "neutron_density_porosity": round(phi, 4),
        "correlation": "RMS neutron-density",
        "inputs": {"nphi": nphi, "dphi": dphi},
        "units": {"neutron_density_porosity": "fraction v/v"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 5. Effective porosity
# ---------------------------------------------------------------------------

def calculate_effective_porosity(
    phi_total: float,
    vshale: float,
) -> str:
    """Calculate effective porosity from total porosity and shale volume.

    Args:
        phi_total: Total porosity (fraction v/v, 0-1).
        vshale: Shale volume (fraction v/v, 0-1).

    Returns:
        JSON string with effective_porosity (fraction v/v).
    """
    _validate_fraction("phi_total", phi_total)
    _validate_fraction("vshale", vshale)

    phie = _clamp(phi_total * (1.0 - vshale))

    return json.dumps({
        "effective_porosity": round(phie, 4),
        "total_porosity": phi_total,
        "vshale": vshale,
        "units": {"effective_porosity": "fraction v/v"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 6. Archie water saturation
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

    from petro_mcp._pro import is_pro
    if not is_pro():
        result["pro_hint"] = (
            "Compute Sw across entire well logs with automated zone picking "
            "in PetroSuite Pro. See petropt.com/pro"
        )

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# 7. Simandoux water saturation
# ---------------------------------------------------------------------------

def calculate_simandoux_sw(
    rt: float,
    phi: float,
    rw: float,
    vshale: float,
    rsh: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
) -> str:
    """Calculate water saturation using the Simandoux equation (shaly sands).

    Modified Simandoux for n=2:
        Sw = C * [sqrt(Vsh^2/Rsh^2 + 4/(C*Rt)) - Vsh/Rsh] / 2
    where C = a * Rw / ((1 - Vsh) * phi^m)

    For arbitrary n, uses bisection solver.

    Args:
        rt: True formation resistivity (ohm-m).
        phi: Porosity (fraction v/v, 0-1).
        rw: Formation water resistivity (ohm-m).
        vshale: Shale volume (fraction v/v, 0-1).
        rsh: Shale resistivity (ohm-m).
        a: Tortuosity factor. Default 1.0.
        m: Cementation exponent. Default 2.0.
        n: Saturation exponent. Default 2.0.

    Returns:
        JSON string with water_saturation (fraction v/v).
    """
    _validate_positive("rt", rt)
    _validate_positive("phi", phi)
    _validate_positive("rw", rw)
    _validate_positive("rsh", rsh)
    _validate_positive("a", a)
    _validate_positive("m", m)
    _validate_positive("n", n)
    if phi > 1:
        raise ValueError(f"phi must be <= 1, got {phi}")
    _validate_fraction("vshale", vshale)

    # For vshale == 0, reduce to Archie
    if vshale == 0:
        return calculate_archie_sw(rt, phi, rw, a, m, n)

    factor = (1.0 - vshale) * phi ** m
    if factor <= 0:
        # Pure shale or zero porosity after shale correction
        return json.dumps({
            "water_saturation": 1.0,
            "hydrocarbon_saturation": 0.0,
            "correlation": "Simandoux (1963)",
            "note": "Non-reservoir: pure shale or zero effective porosity",
            "inputs": {"rt": rt, "phi": phi, "rw": rw, "vshale": vshale,
                       "rsh": rsh, "a": a, "m": m, "n": n},
            "units": {"water_saturation": "fraction v/v"},
        }, indent=2)

    c_val = a * rw / factor

    if n == 2.0:
        # Closed-form quadratic solution
        b_coeff = vshale / rsh
        disc = b_coeff ** 2 + 4.0 / (c_val * rt)
        if disc < 0:
            sw = 1.0
        else:
            sw = c_val * (math.sqrt(disc) - b_coeff) / 2.0
    else:
        # Bisection solver for arbitrary n
        sw = _bisect_simandoux(rt, phi, rw, vshale, rsh, a, m, n)

    sw = _clamp(sw)

    return json.dumps({
        "water_saturation": round(sw, 4),
        "hydrocarbon_saturation": round(1.0 - sw, 4),
        "correlation": "Simandoux (1963)",
        "inputs": {"rt": rt, "phi": phi, "rw": rw, "vshale": vshale,
                   "rsh": rsh, "a": a, "m": m, "n": n},
        "units": {"water_saturation": "fraction v/v"},
    }, indent=2)


def _bisect_simandoux(
    rt: float, phi: float, rw: float,
    vshale: float, rsh: float,
    a: float, m: float, n: float,
) -> float:
    """Solve Simandoux equation by bisection for arbitrary n."""
    def residual(sw: float) -> float:
        factor = (1.0 - vshale) * phi ** m
        return (sw ** n / (a * rw)) * factor + vshale * sw / rsh - 1.0 / rt

    lo, hi = 0.001, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if residual(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# 8. Indonesian water saturation
# ---------------------------------------------------------------------------

def calculate_indonesian_sw(
    rt: float,
    phi: float,
    rw: float,
    vshale: float,
    rsh: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
) -> str:
    """Calculate water saturation using the Indonesian equation.

    Poupon and Leveaux (1971). Designed for high-Vshale formations.

    For n=2, uses closed-form quadratic. For arbitrary n, uses bisection.

    Args:
        rt: True formation resistivity (ohm-m).
        phi: Porosity (fraction v/v, 0-1).
        rw: Formation water resistivity (ohm-m).
        vshale: Shale volume (fraction v/v, 0-1).
        rsh: Shale resistivity (ohm-m).
        a: Tortuosity factor. Default 1.0.
        m: Cementation exponent. Default 2.0.
        n: Saturation exponent. Default 2.0.

    Returns:
        JSON string with water_saturation (fraction v/v).
    """
    _validate_positive("rt", rt)
    _validate_positive("phi", phi)
    _validate_positive("rw", rw)
    _validate_positive("rsh", rsh)
    _validate_positive("a", a)
    _validate_positive("m", m)
    _validate_positive("n", n)
    if phi > 1:
        raise ValueError(f"phi must be <= 1, got {phi}")
    _validate_fraction("vshale", vshale)

    if vshale == 0:
        return calculate_archie_sw(rt, phi, rw, a, m, n)

    if n == 2.0:
        # Closed-form: 1/sqrt(Rt) = A*Sw + B*Sw  where
        # A = phi^(m/2) / sqrt(a*Rw), B = Vsh^(1-Vsh/2) / sqrt(Rsh)
        a_term = phi ** (m / 2.0) / math.sqrt(a * rw)
        b_term = vshale ** (1.0 - vshale / 2.0) / math.sqrt(rsh)
        total = a_term + b_term
        if total <= 0:
            sw = 1.0
        else:
            sw = (1.0 / (math.sqrt(rt) * total))
    else:
        sw = _bisect_indonesian(rt, phi, rw, vshale, rsh, a, m, n)

    sw = _clamp(sw)

    return json.dumps({
        "water_saturation": round(sw, 4),
        "hydrocarbon_saturation": round(1.0 - sw, 4),
        "correlation": "Poupon and Leveaux (1971)",
        "inputs": {"rt": rt, "phi": phi, "rw": rw, "vshale": vshale,
                   "rsh": rsh, "a": a, "m": m, "n": n},
        "units": {"water_saturation": "fraction v/v"},
    }, indent=2)


def _bisect_indonesian(
    rt: float, phi: float, rw: float,
    vshale: float, rsh: float,
    a: float, m: float, n: float,
) -> float:
    """Solve Indonesian equation by bisection for arbitrary n."""
    # Indonesian: (1/sqrt(Rt)) = [Vsh^(1-Vsh/2)/sqrt(Rsh) + phi^(m/2)/sqrt(a*Rw)] * Sw^(n/2)
    a_const = phi ** (m / 2.0) / math.sqrt(a * rw)
    b_const = vshale ** (1.0 - vshale / 2.0) / math.sqrt(rsh)
    bracket = a_const + b_const

    def residual(sw: float) -> float:
        return (bracket * sw ** (n / 2.0)) ** 2 - 1.0 / rt

    lo, hi = 0.001, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if residual(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# 9. Permeability — Timur
# ---------------------------------------------------------------------------

def calculate_permeability_timur(
    phi: float,
    swirr: float,
) -> str:
    """Estimate permeability using Timur (1968) equation.

    k = 0.136 * phi^4.4 / Swirr^2

    Args:
        phi: Porosity (fraction v/v, 0-1).
        swirr: Irreducible water saturation (fraction v/v, 0-1).

    Returns:
        JSON string with permeability in millidarcies.
    """
    _validate_positive("phi", phi)
    _validate_positive("swirr", swirr)
    if phi > 1:
        raise ValueError(f"phi must be <= 1, got {phi}")
    if swirr > 1:
        raise ValueError(f"swirr must be <= 1, got {swirr}")

    # Timur constant 0.136 expects phi and Swirr in percent
    phi_pct = phi * 100.0
    swirr_pct = swirr * 100.0
    k = 0.136 * phi_pct ** 4.4 / swirr_pct ** 2

    return json.dumps({
        "permeability_md": round(k, 4),
        "correlation": "Timur (1968)",
        "inputs": {"phi": phi, "swirr": swirr},
        "units": {"permeability": "mD"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 10. Permeability — Coates
# ---------------------------------------------------------------------------

def calculate_permeability_coates(
    phi: float,
    bvi: float,
    ffi: float,
    c: float = 10.0,
) -> str:
    """Estimate permeability using Coates (1991) equation.

    k = ((phi / C)^2 * (FFI / BVI))^2

    Args:
        phi: Porosity (fraction v/v, 0-1).
        bvi: Bound volume irreducible (fraction v/v).
        ffi: Free fluid index (fraction v/v).
        c: Coates constant. Default 10.0.

    Returns:
        JSON string with permeability in millidarcies.
    """
    _validate_positive("phi", phi)
    _validate_positive("bvi", bvi)
    _validate_non_negative("ffi", ffi)
    _validate_positive("c", c)
    if phi > 1:
        raise ValueError(f"phi must be <= 1, got {phi}")

    if ffi == 0:
        k = 0.0
    else:
        # Coates constant C expects phi in percent when C=10
        phi_pct = phi * 100.0
        k = ((phi_pct / c) ** 2 * (ffi / bvi)) ** 2

    return json.dumps({
        "permeability_md": round(k, 4),
        "correlation": "Coates (1991)",
        "inputs": {"phi": phi, "bvi": bvi, "ffi": ffi, "c": c},
        "units": {"permeability": "mD"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 11. Net pay
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

    from petro_mcp._pro import is_pro
    if not is_pro():
        result["pro_hint"] = (
            "PetroSuite Pro offers multi-well log analysis with automated zone "
            "picking and batch net-pay summaries. See petropt.com/pro"
        )
        result["workspace_hint"] = "Save this analysis to your Petropt workspace: https://tools.petropt.com"

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# 12. Hydrocarbon pore thickness
# ---------------------------------------------------------------------------

def calculate_hpt(
    thickness: float,
    phi: float,
    sw: float,
    ntg: float = 1.0,
) -> str:
    """Calculate hydrocarbon pore thickness (HPT).

    HPT = h * phi * (1 - Sw) * NTG

    Args:
        thickness: Net or gross thickness (ft).
        phi: Average porosity (fraction v/v, 0-1).
        sw: Average water saturation (fraction v/v, 0-1).
        ntg: Net-to-gross ratio (0-1). Default 1.0.

    Returns:
        JSON string with HPT in feet.
    """
    _validate_positive("thickness", thickness)
    _validate_fraction("phi", phi)
    _validate_fraction("sw", sw)
    _validate_fraction("ntg", ntg)

    hpt = thickness * phi * (1.0 - sw) * ntg

    return json.dumps({
        "hpt_ft": round(hpt, 4),
        "correlation": "Hydrocarbon pore thickness",
        "inputs": {
            "thickness": thickness,
            "phi": phi,
            "sw": sw,
            "ntg": ntg,
        },
        "units": {"hpt": "ft"},
    }, indent=2)
