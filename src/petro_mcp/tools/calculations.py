"""Common petroleum engineering calculations for the petro-mcp server."""

from __future__ import annotations

import json
import math


def calculate_ip_ratio(
    oil_rate: float,
    gas_rate: float,
    water_rate: float,
) -> str:
    """Calculate producing ratios: GOR, WOR, water cut, total liquid rate.

    Args:
        oil_rate: Oil production rate in bbl/day (BOPD).
        gas_rate: Gas production rate in Mcf/day.
        water_rate: Water production rate in bbl/day (BWPD).

    Returns:
        JSON string with calculated ratios.
    """
    if oil_rate < 0 or gas_rate < 0 or water_rate < 0:
        raise ValueError("Production rates must be non-negative")

    total_liquid = oil_rate + water_rate

    result = {
        "oil_rate_bopd": oil_rate,
        "gas_rate_mcfd": gas_rate,
        "water_rate_bwpd": water_rate,
        "total_liquid_bfpd": round(total_liquid, 2),
    }

    if oil_rate > 0:
        result["gor_scf_bbl"] = round(gas_rate * 1000 / oil_rate, 1)
        result["wor"] = round(water_rate / oil_rate, 3)
    else:
        result["gor_scf_bbl"] = None
        result["wor"] = None

    if total_liquid > 0:
        result["water_cut_pct"] = round(water_rate / total_liquid * 100, 2)
        result["oil_cut_pct"] = round(oil_rate / total_liquid * 100, 2)
    else:
        result["water_cut_pct"] = 0.0
        result["oil_cut_pct"] = 0.0

    # Classify the well
    if oil_rate > 0 and gas_rate > 0:
        gor = gas_rate * 1000 / oil_rate
        if gor > 100000:
            result["well_type"] = "dry gas"
        elif gor > 20000:
            result["well_type"] = "wet gas / condensate"
        elif gor > 3000:
            result["well_type"] = "volatile oil"
        else:
            result["well_type"] = "black oil"

    return json.dumps(result, indent=2)


def nodal_analysis(
    reservoir_pressure: float,
    PI: float,
    tubing_size: float,
    wellhead_pressure: float,
    depth: float = 8000.0,
    fluid_gradient: float = 0.35,
    num_points: int = 20,
) -> str:
    """Simplified nodal analysis: find IPR/VLP intersection for operating point.

    Uses Vogel IPR (below bubble point) and a simplified vertical lift model.

    Args:
        reservoir_pressure: Average reservoir pressure in psi.
        PI: Productivity index in bbl/day/psi.
        tubing_size: Tubing inner diameter in inches.
        wellhead_pressure: Wellhead flowing pressure in psi.
        depth: True vertical depth in feet.
        fluid_gradient: Fluid pressure gradient in psi/ft.
        num_points: Number of points for the curves.

    Returns:
        JSON string with IPR curve, VLP curve, and operating point.
    """
    if reservoir_pressure <= 0 or PI <= 0:
        raise ValueError("Reservoir pressure and PI must be positive")
    if tubing_size <= 0 or wellhead_pressure < 0:
        raise ValueError("Tubing size must be positive, wellhead pressure non-negative")

    # AOF (absolute open flow) for Vogel IPR
    q_max = PI * reservoir_pressure / 1.8

    rates = [q_max * i / num_points for i in range(num_points + 1)]

    # IPR curve (Vogel equation)
    ipr_pressures = []
    for q in rates:
        if q_max > 0:
            ratio = q / q_max
            # Vogel: Pwf/Pr = 0.125 * (-1 + sqrt(81 - 80*(q/qmax)))
            discriminant = 81 - 80 * ratio
            if discriminant >= 0:
                pwf = reservoir_pressure * 0.125 * (-1 + math.sqrt(discriminant))
            else:
                pwf = 0
        else:
            pwf = reservoir_pressure
        ipr_pressures.append(round(max(pwf, 0), 1))

    # Simplified VLP (vertical lift performance)
    # Pwf = Pwh + hydrostatic_head + friction_loss
    hydrostatic = depth * fluid_gradient
    vlp_pressures = []
    for q in rates:
        # Simplified friction using Hazen-Williams approximation
        # dP_friction proportional to q^1.85 / d^4.87
        if tubing_size > 0 and q > 0:
            # Simplified friction approximation -- not a published petroleum
            # engineering correlation. Provides screening-level results only.
            friction = 0.00005 * (q ** 1.85) / (tubing_size ** 4.87) * depth / 1000
        else:
            friction = 0
        pwf_vlp = wellhead_pressure + hydrostatic + friction
        vlp_pressures.append(round(pwf_vlp, 1))

    # Find intersection (operating point)
    op_rate = None
    op_pressure = None
    for i in range(len(rates) - 1):
        ipr_diff_1 = ipr_pressures[i] - vlp_pressures[i]
        ipr_diff_2 = ipr_pressures[i + 1] - vlp_pressures[i + 1]
        if ipr_diff_1 * ipr_diff_2 <= 0 and ipr_diff_1 != ipr_diff_2:
            # Linear interpolation
            frac = ipr_diff_1 / (ipr_diff_1 - ipr_diff_2)
            op_rate = round(rates[i] + frac * (rates[i + 1] - rates[i]), 1)
            op_pressure = round(ipr_pressures[i] + frac * (ipr_pressures[i + 1] - ipr_pressures[i]), 1)
            break

    result = {
        "reservoir_pressure_psi": reservoir_pressure,
        "productivity_index": PI,
        "tubing_id_inches": tubing_size,
        "wellhead_pressure_psi": wellhead_pressure,
        "depth_ft": depth,
        "aof_bopd": round(q_max, 1),
        "ipr_curve": {
            "rates_bopd": [round(r, 1) for r in rates],
            "pressures_psi": ipr_pressures,
        },
        "vlp_curve": {
            "rates_bopd": [round(r, 1) for r in rates],
            "pressures_psi": vlp_pressures,
        },
    }

    if op_rate is not None:
        result["operating_point"] = {
            "rate_bopd": op_rate,
            "flowing_pressure_psi": op_pressure,
            "drawdown_psi": round(reservoir_pressure - op_pressure, 1),
        }
    else:
        result["operating_point"] = None
        result["note"] = "No intersection found. Well may not flow naturally (consider artificial lift)."

    return json.dumps(result, indent=2)
