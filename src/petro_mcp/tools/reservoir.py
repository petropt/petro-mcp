"""Material balance and volumetric reservoir engineering calculations.

Implements standard reservoir engineering hand calculations:
    - Gas material balance (P/Z vs Gp analysis)
    - Volumetric OOIP and OGIP
    - Recovery factor

References:
    Craft, B.C., Hawkins, M.F., and Terry, R.E., "Applied Petroleum
        Reservoir Engineering," 3rd ed., Prentice Hall, 2015.
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


def _validate_equal_length(names: list[str], *arrays: list) -> None:
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) != 1:
        pairs = ", ".join(f"{n}={l}" for n, l in zip(names, lengths))
        raise ValueError(f"All input arrays must have equal length: {pairs}")


# ---------------------------------------------------------------------------
# 1. Gas Material Balance — P/Z Analysis
# ---------------------------------------------------------------------------

def calculate_pz_analysis(
    pressures: list[float],
    cumulative_gas: list[float],
    abandonment_pressure: float | None = None,
) -> str:
    """P/Z vs cumulative gas production analysis for gas reservoirs.

    Fits a linear trend to P/Z vs Gp data. The x-intercept gives OGIP.

    Args:
        pressures: Reservoir pressures in psi (must include initial pressure
            at Gp=0 or early production data).
        cumulative_gas: Cumulative gas production in Bcf at each pressure.
        abandonment_pressure: Abandonment pressure in psi (optional).
            If provided, calculates recoverable gas.

    Returns:
        JSON string with OGIP, recovery factor, fitted line, R-squared.
    """
    if len(pressures) < 2:
        raise ValueError("At least 2 pressure/Gp data points are required")
    _validate_equal_length(["pressures", "cumulative_gas"], pressures, cumulative_gas)

    for i, p in enumerate(pressures):
        if p <= 0:
            raise ValueError(f"pressures[{i}] must be positive, got {p}")
    for i, g in enumerate(cumulative_gas):
        if g < 0:
            raise ValueError(f"cumulative_gas[{i}] must be non-negative, got {g}")

    if abandonment_pressure is not None and abandonment_pressure <= 0:
        raise ValueError("abandonment_pressure must be positive")

    # Calculate P/Z values.  For this simplified analysis we approximate
    # Z ≈ 1 (ideal gas) unless the caller passes P/Z directly.  The standard
    # reservoir engineering practice is to plot (P/Z) vs Gp where Z comes
    # from a Z-factor correlation, but many practitioners pass pressures
    # that have already been divided by Z.  We document this clearly.
    #
    # Here: pz_values = pressures (caller is expected to supply P/Z or
    # understand the Z=1 assumption).  We label the axis "P/Z".
    pz_values = list(pressures)
    gp_values = list(cumulative_gas)

    # Linear regression: P/Z = slope * Gp + intercept
    n = len(gp_values)
    sum_x = sum(gp_values)
    sum_y = sum(pz_values)
    sum_xy = sum(x * y for x, y in zip(gp_values, pz_values))
    sum_x2 = sum(x * x for x in gp_values)

    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        raise ValueError(
            "Cannot fit linear trend — all cumulative gas values are identical"
        )

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # R-squared
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in pz_values)
    ss_res = sum(
        (y - (slope * x + intercept)) ** 2
        for x, y in zip(gp_values, pz_values)
    )
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # OGIP = x-intercept = -intercept / slope
    if slope >= 0:
        # Slope should be negative for a depleting gas reservoir.
        ogip = None
        ogip_note = (
            "P/Z trend has non-negative slope — data may not represent "
            "a volumetric depletion-drive gas reservoir."
        )
    else:
        ogip = round(-intercept / slope, 4)
        ogip_note = None

    # Initial P/Z (from the fitted line at Gp=0)
    pz_initial = round(intercept, 2)

    # Current recovery factor
    max_gp = max(gp_values)
    recovery_factor = round(max_gp / ogip, 4) if ogip and ogip > 0 else None

    # Abandonment recovery
    abandonment_recovery = None
    recoverable_gas = None
    if abandonment_pressure is not None and ogip is not None and slope != 0:
        # Gp at abandonment = (abandonment_pressure - intercept) / slope
        gp_abandon = (abandonment_pressure - intercept) / slope
        if gp_abandon > 0:
            recoverable_gas = round(gp_abandon, 4)
            abandonment_recovery = round(gp_abandon / ogip, 4)

    # Fitted line data for plotting
    fitted_pz = [round(slope * x + intercept, 2) for x in gp_values]

    result: dict = {
        "method": "P/Z vs Gp (Gas Material Balance)",
        "inputs": {
            "num_data_points": n,
            "pressure_range_psi": [round(min(pressures), 2), round(max(pressures), 2)],
            "gp_range_bcf": [round(min(gp_values), 4), round(max(gp_values), 4)],
        },
        "results": {
            "ogip_bcf": ogip,
            "initial_pz_psi": pz_initial,
            "current_recovery_factor": recovery_factor,
            "r_squared": round(r_squared, 6),
            "slope": round(slope, 6),
            "intercept_psi": round(intercept, 2),
        },
        "units": {
            "ogip": "Bcf",
            "pressures": "psi (or P/Z in psi if Z-corrected)",
            "cumulative_gas": "Bcf",
        },
        "fitted_line": {
            "gp_bcf": [round(g, 4) for g in gp_values],
            "pz_fitted_psi": fitted_pz,
        },
    }

    if ogip_note:
        result["warning"] = ogip_note

    if abandonment_pressure is not None:
        result["abandonment"] = {
            "abandonment_pressure_psi": round(abandonment_pressure, 2),
            "recoverable_gas_bcf": recoverable_gas,
            "abandonment_recovery_factor": abandonment_recovery,
        }


    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# 2. Volumetric OOIP
# ---------------------------------------------------------------------------

def calculate_volumetric_ooip(
    area_acres: float,
    thickness_ft: float,
    porosity: float,
    sw: float,
    bo: float,
) -> str:
    """Calculate original oil in place using the volumetric method.

    OOIP = 7758 * A * h * phi * (1 - Sw) / Bo

    Args:
        area_acres: Reservoir area in acres.
        thickness_ft: Net pay thickness in feet.
        porosity: Porosity (fraction, 0-1).
        sw: Water saturation (fraction, 0-1).
        bo: Oil formation volume factor (bbl/STB).

    Returns:
        JSON string with OOIP in STB.
    """
    _validate_positive("area_acres", area_acres)
    _validate_positive("thickness_ft", thickness_ft)
    _validate_fraction("porosity", porosity)
    _validate_fraction("sw", sw)
    _validate_positive("bo", bo)

    # 7758 bbl/acre-ft is the conversion factor
    bulk_volume_acre_ft = area_acres * thickness_ft
    pore_volume_bbl = 7758.0 * bulk_volume_acre_ft * porosity
    hc_pore_volume_bbl = pore_volume_bbl * (1.0 - sw)
    ooip_stb = hc_pore_volume_bbl / bo

    result = {
        "method": "Volumetric OOIP",
        "inputs": {
            "area_acres": area_acres,
            "thickness_ft": thickness_ft,
            "porosity": porosity,
            "water_saturation": sw,
            "bo_bbl_stb": bo,
        },
        "results": {
            "bulk_volume_acre_ft": round(bulk_volume_acre_ft, 2),
            "pore_volume_bbl": round(pore_volume_bbl, 2),
            "hc_pore_volume_bbl": round(hc_pore_volume_bbl, 2),
            "ooip_stb": round(ooip_stb, 2),
            "ooip_mstb": round(ooip_stb / 1000, 2),
            "ooip_mmstb": round(ooip_stb / 1e6, 4),
        },
        "units": {
            "ooip": "STB",
            "conversion_factor": "7758 bbl/acre-ft",
        },
    }

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# 3. Volumetric OGIP
# ---------------------------------------------------------------------------

def calculate_volumetric_ogip(
    area_acres: float,
    thickness_ft: float,
    porosity: float,
    sw: float,
    bg: float,
) -> str:
    """Calculate original gas in place using the volumetric method.

    OGIP = 43560 * A * h * phi * (1 - Sw) / Bg

    Args:
        area_acres: Reservoir area in acres.
        thickness_ft: Net pay thickness in feet.
        porosity: Porosity (fraction, 0-1).
        sw: Water saturation (fraction, 0-1).
        bg: Gas formation volume factor (ft³/scf).

    Returns:
        JSON string with OGIP in scf.
    """
    _validate_positive("area_acres", area_acres)
    _validate_positive("thickness_ft", thickness_ft)
    _validate_fraction("porosity", porosity)
    _validate_fraction("sw", sw)
    _validate_positive("bg", bg)

    # 43560 ft³/acre-ft
    bulk_volume_acre_ft = area_acres * thickness_ft
    pore_volume_ft3 = 43560.0 * bulk_volume_acre_ft * porosity
    hc_pore_volume_ft3 = pore_volume_ft3 * (1.0 - sw)
    ogip_scf = hc_pore_volume_ft3 / bg

    result = {
        "method": "Volumetric OGIP",
        "inputs": {
            "area_acres": area_acres,
            "thickness_ft": thickness_ft,
            "porosity": porosity,
            "water_saturation": sw,
            "bg_ft3_scf": bg,
        },
        "results": {
            "bulk_volume_acre_ft": round(bulk_volume_acre_ft, 2),
            "pore_volume_ft3": round(pore_volume_ft3, 2),
            "hc_pore_volume_ft3": round(hc_pore_volume_ft3, 2),
            "ogip_scf": round(ogip_scf, 2),
            "ogip_mmscf": round(ogip_scf / 1e6, 4),
            "ogip_bcf": round(ogip_scf / 1e9, 6),
        },
        "units": {
            "ogip": "scf",
            "conversion_factor": "43560 ft³/acre-ft",
        },
    }

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# 4. Recovery Factor
# ---------------------------------------------------------------------------

def calculate_recovery_factor(
    ooip_or_ogip: float,
    cumulative_production: float,
) -> str:
    """Calculate recovery factor as cumulative production / original in place.

    Works for both oil (RF = Np/N) and gas (RF = Gp/G).

    Args:
        ooip_or_ogip: Original oil or gas in place (any consistent unit).
        cumulative_production: Cumulative production (same unit as OOIP/OGIP).

    Returns:
        JSON string with recovery factor (fraction and percent).
    """
    _validate_positive("ooip_or_ogip", ooip_or_ogip)
    _validate_non_negative("cumulative_production", cumulative_production)

    rf = cumulative_production / ooip_or_ogip
    remaining = ooip_or_ogip - cumulative_production

    result = {
        "method": "Recovery Factor",
        "inputs": {
            "original_in_place": ooip_or_ogip,
            "cumulative_production": cumulative_production,
        },
        "results": {
            "recovery_factor": round(rf, 6),
            "recovery_factor_pct": round(rf * 100, 4),
            "remaining_in_place": round(remaining, 4),
        },
        "units": {
            "note": "Units match caller input (STB for oil, scf/Bcf for gas)",
        },
    }

    return json.dumps(result, indent=2)
