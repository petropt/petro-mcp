"""Drilling engineering calculations.

Standard rig-site formulae used daily:
    - Hydrostatic pressure
    - Equivalent circulating density (ECD)
    - Formation pressure gradient
    - Kill mud weight
    - ICP / FCP (Driller's method)
    - MAASP
    - Annular velocity
    - Nozzle total flow area (TFA)
    - Bit pressure drop
    - Burst pressure (Barlow)
    - Collapse pressure (API 5C3)
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


# ---------------------------------------------------------------------------
# 1. Hydrostatic pressure
# ---------------------------------------------------------------------------

def calculate_hydrostatic_pressure(
    mud_weight_ppg: float,
    tvd_ft: float,
) -> str:
    """Calculate hydrostatic pressure from mud weight and TVD.

    P = 0.052 * MW * TVD

    Args:
        mud_weight_ppg: Mud weight in pounds per gallon (ppg).
        tvd_ft: True vertical depth in feet.

    Returns:
        JSON string with hydrostatic pressure in psi.
    """
    _validate_positive("mud_weight_ppg", mud_weight_ppg)
    _validate_positive("tvd_ft", tvd_ft)

    pressure = 0.052 * mud_weight_ppg * tvd_ft

    return json.dumps({
        "hydrostatic_pressure_psi": round(pressure, 2),
        "formula": "P = 0.052 * MW * TVD",
        "inputs": {
            "mud_weight_ppg": mud_weight_ppg,
            "tvd_ft": tvd_ft,
        },
        "units": {"hydrostatic_pressure": "psi", "mud_weight": "ppg", "tvd": "ft"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 2. Equivalent Circulating Density
# ---------------------------------------------------------------------------

def calculate_ecd(
    mud_weight_ppg: float,
    annular_pressure_loss_psi: float,
    tvd_ft: float,
) -> str:
    """Calculate equivalent circulating density (ECD).

    ECD = MW + APL / (0.052 * TVD)

    Args:
        mud_weight_ppg: Static mud weight (ppg).
        annular_pressure_loss_psi: Annular pressure loss (psi).
        tvd_ft: True vertical depth (ft).

    Returns:
        JSON string with ECD in ppg.
    """
    _validate_positive("mud_weight_ppg", mud_weight_ppg)
    _validate_non_negative("annular_pressure_loss_psi", annular_pressure_loss_psi)
    _validate_positive("tvd_ft", tvd_ft)

    ecd = mud_weight_ppg + annular_pressure_loss_psi / (0.052 * tvd_ft)

    return json.dumps({
        "ecd_ppg": round(ecd, 4),
        "formula": "ECD = MW + APL / (0.052 * TVD)",
        "inputs": {
            "mud_weight_ppg": mud_weight_ppg,
            "annular_pressure_loss_psi": annular_pressure_loss_psi,
            "tvd_ft": tvd_ft,
        },
        "units": {"ecd": "ppg"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 3. Formation pressure gradient
# ---------------------------------------------------------------------------

def calculate_formation_pressure_gradient(
    pressure_psi: float,
    tvd_ft: float,
) -> str:
    """Calculate formation pressure gradient as ppg equivalent.

    FPG = P / (0.052 * TVD)

    Args:
        pressure_psi: Formation pressure (psi).
        tvd_ft: True vertical depth (ft).

    Returns:
        JSON string with formation pressure gradient in ppg equivalent.
    """
    _validate_non_negative("pressure_psi", pressure_psi)
    _validate_positive("tvd_ft", tvd_ft)

    fpg = pressure_psi / (0.052 * tvd_ft)

    return json.dumps({
        "formation_pressure_gradient_ppg": round(fpg, 4),
        "formula": "FPG = P / (0.052 * TVD)",
        "inputs": {
            "pressure_psi": pressure_psi,
            "tvd_ft": tvd_ft,
        },
        "units": {"formation_pressure_gradient": "ppg equivalent"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 4. Kill mud weight
# ---------------------------------------------------------------------------

def calculate_kill_mud_weight(
    sidp_psi: float,
    original_mud_weight_ppg: float,
    tvd_ft: float,
) -> str:
    """Calculate kill mud weight for well control.

    Kill MW = Original MW + SIDP / (0.052 * TVD)

    Args:
        sidp_psi: Shut-in drill pipe pressure (psi).
        original_mud_weight_ppg: Original (current) mud weight (ppg).
        tvd_ft: True vertical depth (ft).

    Returns:
        JSON string with kill mud weight in ppg.
    """
    _validate_non_negative("sidp_psi", sidp_psi)
    _validate_positive("original_mud_weight_ppg", original_mud_weight_ppg)
    _validate_positive("tvd_ft", tvd_ft)

    kill_mw = original_mud_weight_ppg + sidp_psi / (0.052 * tvd_ft)

    result = {
        "kill_mud_weight_ppg": round(kill_mw, 4),
        "formula": "Kill MW = Original MW + SIDP / (0.052 * TVD)",
        "inputs": {
            "sidp_psi": sidp_psi,
            "original_mud_weight_ppg": original_mud_weight_ppg,
            "tvd_ft": tvd_ft,
        },
        "units": {"kill_mud_weight": "ppg"},
    }

    from petro_mcp._pro import is_pro
    if not is_pro():
        result["pro_hint"] = (
            "Petropt Pro includes real-time well control simulations "
            "and integrated well planning. See tools.petropt.com/pricing"
        )

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# 5. ICP and FCP (Driller's method / Wait-and-Weight)
# ---------------------------------------------------------------------------

def calculate_icp_fcp(
    sidp_psi: float,
    circulating_pressure_psi: float,
    kill_mud_weight_ppg: float,
    original_mud_weight_ppg: float,
) -> str:
    """Calculate Initial Circulating Pressure and Final Circulating Pressure.

    ICP = SIDP + slow circulating pressure (SCP)
    FCP = SCP * (Kill MW / Original MW)

    Args:
        sidp_psi: Shut-in drill pipe pressure (psi).
        circulating_pressure_psi: Slow circulating pressure / SCP (psi).
        kill_mud_weight_ppg: Kill mud weight (ppg).
        original_mud_weight_ppg: Original mud weight (ppg).

    Returns:
        JSON string with ICP and FCP in psi.
    """
    _validate_non_negative("sidp_psi", sidp_psi)
    _validate_positive("circulating_pressure_psi", circulating_pressure_psi)
    _validate_positive("kill_mud_weight_ppg", kill_mud_weight_ppg)
    _validate_positive("original_mud_weight_ppg", original_mud_weight_ppg)

    icp = sidp_psi + circulating_pressure_psi
    fcp = circulating_pressure_psi * (kill_mud_weight_ppg / original_mud_weight_ppg)

    return json.dumps({
        "icp_psi": round(icp, 2),
        "fcp_psi": round(fcp, 2),
        "formula": {
            "icp": "ICP = SIDP + SCP",
            "fcp": "FCP = SCP * (Kill MW / Original MW)",
        },
        "inputs": {
            "sidp_psi": sidp_psi,
            "circulating_pressure_psi": circulating_pressure_psi,
            "kill_mud_weight_ppg": kill_mud_weight_ppg,
            "original_mud_weight_ppg": original_mud_weight_ppg,
        },
        "units": {"icp": "psi", "fcp": "psi"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 6. MAASP
# ---------------------------------------------------------------------------

def calculate_maasp(
    fracture_gradient_ppg: float,
    mud_weight_ppg: float,
    shoe_tvd_ft: float,
) -> str:
    """Calculate Maximum Allowable Annular Surface Pressure.

    MAASP = (FG - MW) * 0.052 * shoe TVD

    Args:
        fracture_gradient_ppg: Fracture gradient at shoe (ppg).
        mud_weight_ppg: Current mud weight (ppg).
        shoe_tvd_ft: Casing shoe TVD (ft).

    Returns:
        JSON string with MAASP in psi.
    """
    _validate_positive("fracture_gradient_ppg", fracture_gradient_ppg)
    _validate_positive("mud_weight_ppg", mud_weight_ppg)
    _validate_positive("shoe_tvd_ft", shoe_tvd_ft)

    maasp = (fracture_gradient_ppg - mud_weight_ppg) * 0.052 * shoe_tvd_ft

    return json.dumps({
        "maasp_psi": round(maasp, 2),
        "formula": "MAASP = (FG - MW) * 0.052 * shoe TVD",
        "inputs": {
            "fracture_gradient_ppg": fracture_gradient_ppg,
            "mud_weight_ppg": mud_weight_ppg,
            "shoe_tvd_ft": shoe_tvd_ft,
        },
        "units": {"maasp": "psi"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 7. Annular velocity
# ---------------------------------------------------------------------------

def calculate_annular_velocity(
    flow_rate_gpm: float,
    hole_diameter_in: float,
    pipe_od_in: float,
) -> str:
    """Calculate annular velocity.

    AV = 24.51 * Q / (Dh^2 - Dp^2)

    Args:
        flow_rate_gpm: Flow rate (gallons per minute).
        hole_diameter_in: Hole / casing ID (inches).
        pipe_od_in: Pipe / drill string OD (inches).

    Returns:
        JSON string with annular velocity in ft/min.
    """
    _validate_positive("flow_rate_gpm", flow_rate_gpm)
    _validate_positive("hole_diameter_in", hole_diameter_in)
    _validate_positive("pipe_od_in", pipe_od_in)

    if pipe_od_in >= hole_diameter_in:
        raise ValueError(
            f"pipe_od_in ({pipe_od_in}) must be less than "
            f"hole_diameter_in ({hole_diameter_in})"
        )

    annular_area = hole_diameter_in ** 2 - pipe_od_in ** 2
    av = 24.51 * flow_rate_gpm / annular_area

    return json.dumps({
        "annular_velocity_ft_per_min": round(av, 2),
        "formula": "AV = 24.51 * Q / (Dh^2 - Dp^2)",
        "inputs": {
            "flow_rate_gpm": flow_rate_gpm,
            "hole_diameter_in": hole_diameter_in,
            "pipe_od_in": pipe_od_in,
        },
        "units": {"annular_velocity": "ft/min"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 8. Nozzle TFA
# ---------------------------------------------------------------------------

def calculate_nozzle_tfa(
    nozzle_sizes: list[int],
) -> str:
    """Calculate total flow area (TFA) of bit nozzles.

    TFA = sum(pi/4 * (d/32)^2) for each nozzle size in 32nds of an inch.

    Args:
        nozzle_sizes: List of nozzle sizes in 32nds of an inch (e.g. [12, 12, 12]).

    Returns:
        JSON string with TFA in square inches.
    """
    if not nozzle_sizes:
        raise ValueError("nozzle_sizes must not be empty")

    for i, size in enumerate(nozzle_sizes):
        if size <= 0:
            raise ValueError(f"nozzle_sizes[{i}] must be positive, got {size}")

    tfa = 0.0
    for size in nozzle_sizes:
        diameter_in = size / 32.0
        tfa += math.pi / 4.0 * diameter_in ** 2

    return json.dumps({
        "tfa_sqin": round(tfa, 4),
        "nozzle_count": len(nozzle_sizes),
        "formula": "TFA = sum(pi/4 * (d/32)^2)",
        "inputs": {
            "nozzle_sizes_32nds": nozzle_sizes,
        },
        "units": {"tfa": "in^2"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 9. Bit pressure drop
# ---------------------------------------------------------------------------

def calculate_bit_pressure_drop(
    flow_rate_gpm: float,
    mud_weight_ppg: float,
    tfa_sqin: float,
) -> str:
    """Calculate pressure drop across the bit.

    dP = (MW * Q^2) / (12032 * TFA^2)

    Args:
        flow_rate_gpm: Flow rate (gallons per minute).
        mud_weight_ppg: Mud weight (ppg).
        tfa_sqin: Total flow area of nozzles (in^2).

    Returns:
        JSON string with bit pressure drop in psi.
    """
    _validate_positive("flow_rate_gpm", flow_rate_gpm)
    _validate_positive("mud_weight_ppg", mud_weight_ppg)
    _validate_positive("tfa_sqin", tfa_sqin)

    dp = (mud_weight_ppg * flow_rate_gpm ** 2) / (12032.0 * tfa_sqin ** 2)

    return json.dumps({
        "bit_pressure_drop_psi": round(dp, 2),
        "formula": "dP = (MW * Q^2) / (12032 * TFA^2)",
        "inputs": {
            "flow_rate_gpm": flow_rate_gpm,
            "mud_weight_ppg": mud_weight_ppg,
            "tfa_sqin": tfa_sqin,
        },
        "units": {"bit_pressure_drop": "psi"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 10. Burst pressure (Barlow formula)
# ---------------------------------------------------------------------------

def calculate_burst_pressure(
    yield_strength_psi: float,
    wall_thickness_in: float,
    od_in: float,
) -> str:
    """Calculate internal burst pressure using Barlow's formula with API design factor.

    P_burst = 0.875 * 2 * Fy * t / OD

    The 0.875 factor accounts for the 12.5% minimum wall thickness tolerance (API).

    Args:
        yield_strength_psi: Minimum yield strength of the pipe (psi).
        wall_thickness_in: Nominal wall thickness (inches).
        od_in: Outer diameter (inches).

    Returns:
        JSON string with burst pressure in psi.
    """
    _validate_positive("yield_strength_psi", yield_strength_psi)
    _validate_positive("wall_thickness_in", wall_thickness_in)
    _validate_positive("od_in", od_in)

    if wall_thickness_in >= od_in / 2.0:
        raise ValueError(
            f"wall_thickness_in ({wall_thickness_in}) must be less than "
            f"od_in / 2 ({od_in / 2.0})"
        )

    p_burst = 0.875 * 2.0 * yield_strength_psi * wall_thickness_in / od_in

    return json.dumps({
        "burst_pressure_psi": round(p_burst, 2),
        "formula": "Barlow: P_burst = 0.875 * 2 * Fy * t / OD",
        "inputs": {
            "yield_strength_psi": yield_strength_psi,
            "wall_thickness_in": wall_thickness_in,
            "od_in": od_in,
        },
        "units": {"burst_pressure": "psi"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 11. Collapse pressure (API 5C3)
# ---------------------------------------------------------------------------

# API 5C3 empirical coefficients by D/t regime:
#   Yield collapse:    P = 2*Fy * [(D/t - 1) / (D/t)^2]
#   Plastic collapse:  P = Fy * [A / (D/t) - B] - C
#   Transition collapse: P = Fy * [F / (D/t) - G]
#   Elastic collapse:  P = 46.95e6 / [(D/t) * (D/t - 1)^2]
#
# The regime boundaries and A, B, C, F, G coefficients depend on grade (yield strength).

def _api5c3_coefficients(fy: float) -> dict:
    """Compute API 5C3 collapse coefficients from yield strength.

    Uses the API 5C3 formulas for A, B, C, F, G and the regime boundary
    D/t ratios. These formulas are from API Bulletin 5C3 (1994).
    """
    # A, B, C from API 5C3 empirical formulas
    a = 2.8762 + 0.10679e-5 * fy + 0.21301e-10 * fy ** 2 - 0.53132e-16 * fy ** 3
    b = 0.026233 + 0.50609e-6 * fy
    c = -465.93 + 0.030867 * fy - 0.10483e-7 * fy ** 2 + 0.36989e-13 * fy ** 3

    # Yield-plastic boundary (D/t)_yp
    # Solve: 2*Fy*[(D/t-1)/(D/t)^2] = Fy*[A/(D/t) - B] - C
    # Iterative: (D/t)_yp = (sqrt((A-2)^2 + 8*(B + C/Fy)) + (A-2)) / (2*(B + C/Fy))
    bc = b + c / fy
    dt_yp = (math.sqrt((a - 2) ** 2 + 8 * bc) + (a - 2)) / (2 * bc)

    # F and G from API 5C3
    f = 46.95e6 * (3 * b / a) / (2 + b / a)
    f = f / (fy * (3 * b / a - (b / a) ** 2))  # noqa: E741
    # Recalculate properly per API 5C3:
    # F = 46.95e6 * [3*B/A / (2 + B/A)] / [Fy * (3*B/A - (B/A)^2)]
    # Simpler form from the standard:
    ba_ratio = b / a
    numerator = 46.95e6 * (3.0 * ba_ratio / (2.0 + ba_ratio))
    denominator = fy * (3.0 * ba_ratio - ba_ratio ** 2)
    f = numerator / denominator

    g = f * b / a

    # Plastic-transition boundary (D/t)_pt
    dt_pt = (fy * (a - f)) / (c + fy * (b - g))

    # Transition-elastic boundary (D/t)_te
    # Solve: Fy*[F/(D/t) - G] = 46.95e6 / [(D/t)*(D/t-1)^2]
    # Use the formula: (D/t)_te = (2 + B/A) / (3*B/A)
    # Actually from the standard:
    dt_te = 2.0 + ba_ratio / (3.0 * ba_ratio)
    # Correct form: find where transition = elastic
    # From API 5C3: (D/t)_te = root of 46.95e6/[(D/t)*(D/t-1)^2] = Fy*[F/(D/t)-G]
    # Use Newton's method
    dt_te = _solve_te_boundary(fy, f, g)

    return {
        "A": a, "B": b, "C": c, "F": f, "G": g,
        "dt_yp": dt_yp, "dt_pt": dt_pt, "dt_te": dt_te,
    }


def _solve_te_boundary(fy: float, f: float, g: float) -> float:
    """Solve for transition-elastic D/t boundary by bisection."""
    def residual(dt: float) -> float:
        transition = fy * (f / dt - g)
        elastic = 46.95e6 / (dt * (dt - 1) ** 2)
        return transition - elastic

    lo, hi = 2.0, 80.0
    # Make sure signs differ
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if residual(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def calculate_collapse_pressure(
    od_in: float,
    wall_thickness_in: float,
    yield_strength_psi: float,
    grade: str = "",
) -> str:
    """Calculate collapse pressure rating per API 5C3.

    Determines the collapse regime (yield, plastic, transition, elastic)
    based on D/t ratio and yield strength, then applies the corresponding
    API 5C3 formula.

    Args:
        od_in: Casing outer diameter (inches).
        wall_thickness_in: Wall thickness (inches).
        yield_strength_psi: Minimum yield strength (psi).
        grade: Optional API grade label (e.g. 'J-55', 'N-80') for reference only.

    Returns:
        JSON string with collapse pressure and regime classification.
    """
    _validate_positive("od_in", od_in)
    _validate_positive("wall_thickness_in", wall_thickness_in)
    _validate_positive("yield_strength_psi", yield_strength_psi)

    if wall_thickness_in >= od_in / 2.0:
        raise ValueError(
            f"wall_thickness_in ({wall_thickness_in}) must be less than "
            f"od_in / 2 ({od_in / 2.0})"
        )

    dt = od_in / wall_thickness_in
    coeff = _api5c3_coefficients(yield_strength_psi)

    fy = yield_strength_psi
    a = coeff["A"]
    b = coeff["B"]
    c = coeff["C"]
    f = coeff["F"]
    g = coeff["G"]

    if dt <= coeff["dt_yp"]:
        # Yield collapse
        regime = "yield"
        p_collapse = 2.0 * fy * ((dt - 1) / dt ** 2)
    elif dt <= coeff["dt_pt"]:
        # Plastic collapse
        regime = "plastic"
        p_collapse = fy * (a / dt - b) - c
    elif dt <= coeff["dt_te"]:
        # Transition collapse
        regime = "transition"
        p_collapse = fy * (f / dt - g)
    else:
        # Elastic collapse
        regime = "elastic"
        p_collapse = 46.95e6 / (dt * (dt - 1) ** 2)

    p_collapse = max(p_collapse, 0.0)

    result: dict = {
        "collapse_pressure_psi": round(p_collapse, 2),
        "regime": regime,
        "d_over_t": round(dt, 4),
        "method": "API 5C3",
        "inputs": {
            "od_in": od_in,
            "wall_thickness_in": wall_thickness_in,
            "yield_strength_psi": yield_strength_psi,
        },
        "units": {"collapse_pressure": "psi"},
    }
    if grade:
        result["inputs"]["grade"] = grade

    from petro_mcp._pro import is_pro
    if not is_pro():
        result["pro_hint"] = (
            "Petropt Pro provides real-time drilling optimization and "
            "casing design with full well planning workflows. See tools.petropt.com/pricing"
        )

    return json.dumps(result, indent=2)
