"""Production engineering calculations for the petro-mcp server.

Multiphase flow, liquid loading, flow assurance, erosional velocity,
and choke performance tools used in daily production operations.
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
# Fluid property helpers (internal)
# ---------------------------------------------------------------------------

def _oil_sg(api: float) -> float:
    """Specific gravity of oil from API gravity."""
    return 141.5 / (131.5 + api)


def _dead_oil_viscosity_beggs(api: float, temp_f: float) -> float:
    """Beggs-Robinson dead oil viscosity (cp)."""
    x = temp_f ** (-1.163) * math.exp(6.9824 - 0.04658 * api)
    return 10.0 ** x - 1.0


def _solution_gor_standing(api: float, gas_sg: float, temp_f: float, pressure: float) -> float:
    """Standing correlation for solution GOR (scf/bbl)."""
    if pressure <= 0:
        return 0.0
    yg = gas_sg
    return yg * ((pressure / 18.2 + 1.4) * 10.0 ** (0.0125 * api - 0.00091 * temp_f)) ** 1.2048


def _oil_fvf_standing(api: float, gas_sg: float, rs: float, temp_f: float) -> float:
    """Standing correlation for oil FVF (bbl/STB)."""
    gamma_o = _oil_sg(api)
    f = rs * (gas_sg / gamma_o) ** 0.5 + 1.25 * temp_f
    return 0.9759 + 0.00012 * f ** 1.2


def _gas_z_factor_simple(temp_f: float, pressure: float, gas_sg: float) -> float:
    """Simplified gas Z-factor using Hall-Yarborough approximation.

    For internal use in pressure traverse calculations.
    """
    pressure = max(pressure, 14.7)  # floor at atmospheric
    tpc = 168.0 + 325.0 * gas_sg - 12.5 * gas_sg ** 2
    ppc = 677.0 + 15.0 * gas_sg - 37.5 * gas_sg ** 2
    tpr = (temp_f + 460.0) / tpc
    ppr = pressure / ppc
    # Simple Brill-Beggs approximation for Z
    a = 1.39 * (tpr - 0.92) ** 0.5 - 0.36 * tpr - 0.101
    b = (0.62 - 0.23 * tpr) * ppr + (0.066 / (tpr - 0.86) - 0.037) * ppr ** 2 + 0.32 * math.exp(-19.53 * (tpr - 1.0)) * ppr ** 6 / (10.0 ** (9.0 * (tpr - 1.0)))
    c = 0.132 - 0.32 * math.log10(tpr)
    d = 10.0 ** (0.3106 - 0.49 * tpr + 0.1824 * tpr ** 2)
    z = a + (1.0 - a) * math.exp(-b) + c * ppr ** d
    return max(z, 0.05)


def _gas_density(pressure: float, temp_f: float, gas_sg: float) -> float:
    """Gas density in lb/ft3."""
    pressure = max(pressure, 14.7)
    z = _gas_z_factor_simple(temp_f, pressure, gas_sg)
    mw = 28.97 * gas_sg
    temp_r = temp_f + 460.0
    return pressure * mw / (z * 10.73 * temp_r)


def _water_density_lb_ft3(water_sg: float) -> float:
    """Water density from specific gravity."""
    return water_sg * 62.4


def _surface_tension_water() -> float:
    """Surface tension of water in dynes/cm (approximate)."""
    return 60.0


def _surface_tension_condensate() -> float:
    """Surface tension of condensate in dynes/cm (approximate)."""
    return 20.0


# ---------------------------------------------------------------------------
# 1. Beggs & Brill Multiphase Pressure Drop
# ---------------------------------------------------------------------------

def _beggs_brill_flow_pattern(nfr: float, lambda_l: float) -> str:
    """Determine Beggs & Brill flow pattern.

    Args:
        nfr: Froude number.
        lambda_l: Input liquid fraction (no-slip holdup).

    Returns:
        Flow pattern string.
    """
    if lambda_l < 0.001:
        return "distributed"

    # Transition boundaries from Beggs & Brill (1973)
    l1 = 316.0 * lambda_l ** 0.302
    l2 = 0.0009252 * lambda_l ** (-2.4684)
    l3 = 0.10 * lambda_l ** (-1.4516)
    l4 = 0.5 * lambda_l ** (-6.738)

    if lambda_l < 0.01 and nfr < l1:
        return "segregated"
    if lambda_l >= 0.01 and nfr < l2:
        return "segregated"

    if lambda_l >= 0.01 and l2 <= nfr <= l3:
        return "transition"

    if (0.01 <= lambda_l < 0.4 and l3 < nfr <= l1) or (lambda_l >= 0.4 and l3 < nfr <= l4):
        return "intermittent"

    return "distributed"


def _beggs_brill_holdup_horizontal(lambda_l: float, nfr: float, pattern: str) -> float:
    """Calculate liquid holdup for horizontal flow (Beggs & Brill)."""
    if lambda_l <= 0.0:
        return 0.0

    # Coefficients for each pattern
    if pattern == "segregated":
        a, b, c = 0.98, 0.4846, 0.0868
    elif pattern == "intermittent":
        a, b, c = 0.845, 0.5351, 0.0173
    elif pattern == "distributed":
        a, b, c = 1.065, 0.5824, 0.0609
    else:
        # For transition, will be interpolated
        a, b, c = 0.845, 0.5351, 0.0173

    hl0 = a * lambda_l ** b / max(nfr ** c, 1e-10)
    return max(hl0, lambda_l)


def _beggs_brill_inclination_correction(
    hl0: float, lambda_l: float, nfr: float, nlv: float,
    inclination_rad: float, pattern: str,
) -> float:
    """Apply inclination correction to horizontal holdup."""
    if pattern == "distributed":
        return hl0  # No inclination correction for distributed

    # C coefficient
    if pattern == "segregated":
        e, f, g, h = 0.011, -3.768, 3.539, -1.614
    elif pattern == "intermittent":
        e, f, g, h = 2.96, 0.305, -0.4473, 0.0978
    else:
        e, f, g, h = 2.96, 0.305, -0.4473, 0.0978

    c_val = max(0.0, (1.0 - lambda_l) * math.log(
        max(e * lambda_l ** f * nlv ** g * nfr ** h, 1e-10)
    ))

    # Psi correction
    psi = 1.0 + c_val * (math.sin(1.8 * inclination_rad) - 0.333 * math.sin(1.8 * inclination_rad) ** 3)
    return max(hl0 * psi, 0.0)


def calculate_beggs_brill_pressure_drop(
    flow_rate_bpd: float,
    gor_scf_bbl: float,
    water_cut: float,
    oil_api: float,
    gas_sg: float,
    pipe_id_in: float,
    pipe_length_ft: float,
    inclination_deg: float,
    wellhead_pressure_psi: float,
    temperature_f: float,
) -> str:
    """Beggs & Brill (1973) multiphase pressure drop calculation.

    The most widely used multiphase flow correlation for oil and gas wells.
    Determines flow pattern, calculates liquid holdup, friction factor,
    and pressure gradient including elevation, friction, and acceleration terms.

    Args:
        flow_rate_bpd: Total liquid flow rate in bbl/day.
        gor_scf_bbl: Gas-oil ratio in scf/bbl.
        water_cut: Water cut as fraction (0-1).
        oil_api: Oil API gravity.
        gas_sg: Gas specific gravity (air = 1.0).
        pipe_id_in: Pipe inner diameter in inches.
        pipe_length_ft: Pipe length in feet.
        inclination_deg: Pipe inclination from horizontal in degrees
            (90 = vertical upward, 0 = horizontal, -90 = downward).
        wellhead_pressure_psi: Wellhead (outlet) pressure in psi.
        temperature_f: Average flowing temperature in degrees F.

    Returns:
        JSON string with pressure drop, BHP, flow pattern, holdup, and details.
    """
    _validate_positive("flow_rate_bpd", flow_rate_bpd)
    _validate_non_negative("gor_scf_bbl", gor_scf_bbl)
    _validate_non_negative("water_cut", water_cut)
    if water_cut > 1.0:
        raise ValueError(f"water_cut must be between 0 and 1, got {water_cut}")
    _validate_positive("oil_api", oil_api)
    _validate_positive("gas_sg", gas_sg)
    _validate_positive("pipe_id_in", pipe_id_in)
    _validate_positive("pipe_length_ft", pipe_length_ft)
    _validate_non_negative("wellhead_pressure_psi", wellhead_pressure_psi)
    _validate_positive("temperature_f", temperature_f)
    if not -90.0 <= inclination_deg <= 90.0:
        raise ValueError(f"inclination_deg must be between -90 and 90, got {inclination_deg}")

    # Average pressure estimate for fluid properties (iterate once)
    p_avg = wellhead_pressure_psi + 500.0  # initial guess

    for _ in range(3):
        # --- Fluid properties at average conditions ---
        oil_rate_bpd = flow_rate_bpd * (1.0 - water_cut)
        water_rate_bpd = flow_rate_bpd * water_cut

        # Solution GOR
        rs = _solution_gor_standing(oil_api, gas_sg, temperature_f, p_avg)
        free_gor = max(gor_scf_bbl - rs, 0.0)

        # Oil FVF
        bo = _oil_fvf_standing(oil_api, gas_sg, min(rs, gor_scf_bbl), temperature_f)
        bw = 1.0 + 1e-6 * (p_avg - 14.7)  # simplified water FVF

        # Gas Z and Bg
        z = _gas_z_factor_simple(temperature_f, p_avg, gas_sg)
        temp_r = temperature_f + 460.0
        bg = 0.0283 * z * temp_r / max(p_avg, 14.7)  # ft3/scf

        # Volumetric flow rates at flowing conditions (ft3/s)
        q_oil = oil_rate_bpd * bo * 5.615 / 86400.0  # ft3/s
        q_water = water_rate_bpd * bw * 5.615 / 86400.0
        q_gas = oil_rate_bpd * free_gor * bg / 86400.0  # ft3/s

        q_liquid = q_oil + q_water
        q_total = q_liquid + q_gas

        # Pipe cross-sectional area
        d_ft = pipe_id_in / 12.0
        area = math.pi / 4.0 * d_ft ** 2

        if q_total <= 0 or area <= 0:
            return json.dumps({
                "error": "No flow at average conditions",
                "inputs": {
                    "flow_rate_bpd": flow_rate_bpd,
                    "wellhead_pressure_psi": wellhead_pressure_psi,
                },
            }, indent=2)

        # Mixture velocities
        v_sl = q_liquid / area  # superficial liquid velocity (ft/s)
        v_sg = q_gas / area     # superficial gas velocity (ft/s)
        v_m = v_sl + v_sg       # mixture velocity

        # Input liquid fraction (no-slip holdup)
        lambda_l = v_sl / v_m if v_m > 0 else 1.0
        lambda_g = 1.0 - lambda_l

        # Fluid densities
        rho_oil = _oil_sg(oil_api) * 62.4 / bo
        rho_water = 62.4 * 1.07 / bw
        rho_gas = _gas_density(p_avg, temperature_f, gas_sg)

        # Liquid density (oil + water mixture)
        if q_liquid > 0:
            f_oil = q_oil / q_liquid
            f_water = q_water / q_liquid
        else:
            f_oil, f_water = 0.5, 0.5
        rho_l = rho_oil * f_oil + rho_water * f_water

        # No-slip density
        rho_ns = rho_l * lambda_l + rho_gas * lambda_g

        # Froude number
        nfr = v_m ** 2 / (32.174 * d_ft) if d_ft > 0 else 0.0

        # Liquid velocity number
        sigma_l = _surface_tension_water() if water_cut > 0.5 else 20.0
        nlv = 1.938 * v_sl * (rho_l / max(sigma_l, 1e-6)) ** 0.25

        # --- Flow pattern determination ---
        pattern = _beggs_brill_flow_pattern(nfr, lambda_l)

        # --- Liquid holdup ---
        inclination_rad = math.radians(inclination_deg)

        if pattern == "transition":
            # Interpolate between segregated and intermittent
            l2 = 0.0009252 * max(lambda_l, 1e-6) ** (-2.4684)
            l3 = 0.10 * max(lambda_l, 1e-6) ** (-1.4516)
            a_factor = (l3 - nfr) / max(l3 - l2, 1e-10)
            a_factor = max(0.0, min(1.0, a_factor))

            hl_seg = _beggs_brill_holdup_horizontal(lambda_l, nfr, "segregated")
            hl_seg = _beggs_brill_inclination_correction(
                hl_seg, lambda_l, nfr, nlv, inclination_rad, "segregated")

            hl_int = _beggs_brill_holdup_horizontal(lambda_l, nfr, "intermittent")
            hl_int = _beggs_brill_inclination_correction(
                hl_int, lambda_l, nfr, nlv, inclination_rad, "intermittent")

            hl = a_factor * hl_seg + (1.0 - a_factor) * hl_int
        else:
            hl = _beggs_brill_holdup_horizontal(lambda_l, nfr, pattern)
            hl = _beggs_brill_inclination_correction(
                hl, lambda_l, nfr, nlv, inclination_rad, pattern)

        hl = max(0.0, min(1.0, hl))

        # Slip (two-phase) density
        rho_s = rho_l * hl + rho_gas * (1.0 - hl)

        # --- Friction factor ---
        # Liquid viscosity
        mu_oil = _dead_oil_viscosity_beggs(oil_api, temperature_f)
        mu_water = 1.0  # cp, approximate
        mu_l = mu_oil * f_oil + mu_water * f_water
        mu_gas = 0.012  # cp, approximate

        mu_ns = mu_l * lambda_l + mu_gas * lambda_g

        # Reynolds number (no-slip)
        re_ns = 1488.0 * rho_ns * v_m * d_ft / max(mu_ns, 1e-10)

        # Moody friction factor (smooth pipe approximation)
        if re_ns < 2000:
            fn = 64.0 / max(re_ns, 1.0)
        else:
            fn = 0.0056 + 0.5 / max(re_ns, 1.0) ** 0.32

        # Two-phase friction factor ratio
        y = lambda_l / max(hl ** 2, 1e-10)
        if 1.0 < y < 1.2:
            s = math.log(2.2 * y - 1.2)
        else:
            ln_y = math.log(max(y, 1e-10))
            s = ln_y / (-0.0523 + 3.182 * ln_y - 0.8725 * ln_y ** 2 + 0.01853 * ln_y ** 4) if abs(ln_y) > 1e-6 else 0.0

        ftp = fn * math.exp(s)

        # --- Pressure gradient ---
        # Elevation component (psi/ft)
        dp_el = rho_s * math.sin(inclination_rad) / 144.0

        # Friction component (psi/ft)
        dp_fric = ftp * rho_ns * v_m ** 2 / (2.0 * 32.174 * d_ft * 144.0)

        # Acceleration (usually small, include for completeness)
        dp_acc = rho_s * v_m * v_sg / (32.174 * p_avg * 144.0)
        ek = rho_s * v_m * v_sg / (32.174 * p_avg * 144.0)

        # Total pressure gradient
        dp_dz = (dp_el + dp_fric) / max(1.0 - ek, 0.01)

        # Total pressure drop
        pressure_drop = dp_dz * pipe_length_ft
        bhp = wellhead_pressure_psi + pressure_drop

        # Update average pressure
        p_avg = max((wellhead_pressure_psi + bhp) / 2.0, 14.7)

    result = {
        "inputs": {
            "flow_rate_bpd": flow_rate_bpd,
            "gor_scf_bbl": gor_scf_bbl,
            "water_cut": water_cut,
            "oil_api": oil_api,
            "gas_sg": gas_sg,
            "pipe_id_in": pipe_id_in,
            "pipe_length_ft": pipe_length_ft,
            "inclination_deg": inclination_deg,
            "wellhead_pressure_psi": wellhead_pressure_psi,
            "temperature_f": temperature_f,
        },
        "correlation": "Beggs and Brill (1973)",
        "units": {
            "pressure": "psi",
            "pressure_gradient": "psi/ft",
            "velocity": "ft/s",
            "density": "lb/ft3",
        },
        "flow_pattern": pattern,
        "liquid_holdup": round(hl, 4),
        "no_slip_holdup": round(lambda_l, 4),
        "mixture_velocity_ft_s": round(v_m, 3),
        "superficial_liquid_velocity_ft_s": round(v_sl, 3),
        "superficial_gas_velocity_ft_s": round(v_sg, 3),
        "two_phase_friction_factor": round(ftp, 6),
        "reynolds_number": round(re_ns, 0),
        "pressure_gradient_psi_ft": round(dp_dz, 6),
        "elevation_gradient_psi_ft": round(dp_el, 6),
        "friction_gradient_psi_ft": round(dp_fric, 6),
        "pressure_drop_psi": round(pressure_drop, 1),
        "flowing_bhp_psi": round(bhp, 1),
        "slip_density_lb_ft3": round(rho_s, 3),
        "no_slip_density_lb_ft3": round(rho_ns, 3),
    }

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# 2. Turner Critical Rate (Liquid Loading)
# ---------------------------------------------------------------------------

def calculate_turner_critical_rate(
    wellhead_pressure_psi: float,
    wellhead_temp_f: float,
    gas_sg: float,
    condensate_sg: float | None = None,
    water_sg: float = 1.07,
    tubing_id_in: float = 2.441,
    current_rate_mcfd: float | None = None,
) -> str:
    """Turner et al. (1969) critical rate for gas well liquid unloading.

    Calculates the minimum gas velocity and flow rate needed to continuously
    remove liquids from a gas well. Uses the droplet model:
    v_critical = 1.593 * sigma^0.25 * (rho_l - rho_g)^0.25 / rho_g^0.5

    Args:
        wellhead_pressure_psi: Wellhead flowing pressure in psi.
        wellhead_temp_f: Wellhead temperature in degrees F.
        gas_sg: Gas specific gravity (air = 1.0).
        condensate_sg: Condensate specific gravity (if present).
        water_sg: Water specific gravity. Default 1.07.
        tubing_id_in: Tubing inner diameter in inches. Default 2.441.
        current_rate_mcfd: Current gas rate in Mcf/d for status check (optional).

    Returns:
        JSON string with critical velocity, critical rate, and loading status.
    """
    _validate_positive("wellhead_pressure_psi", wellhead_pressure_psi)
    _validate_positive("wellhead_temp_f", wellhead_temp_f)
    _validate_positive("gas_sg", gas_sg)
    _validate_positive("tubing_id_in", tubing_id_in)
    if condensate_sg is not None:
        _validate_positive("condensate_sg", condensate_sg)

    # Gas density at wellhead conditions
    rho_g = _gas_density(wellhead_pressure_psi, wellhead_temp_f, gas_sg)

    results = {}

    # Water case (always calculated)
    rho_w = water_sg * 62.4  # lb/ft3
    sigma_w = _surface_tension_water()  # dynes/cm
    v_crit_w = 1.593 * sigma_w ** 0.25 * (rho_w - rho_g) ** 0.25 / rho_g ** 0.5

    # Condensate case
    v_crit_c = None
    if condensate_sg is not None:
        rho_c = condensate_sg * 62.4
        sigma_c = _surface_tension_condensate()
        v_crit_c = 1.593 * sigma_c ** 0.25 * (rho_c - rho_g) ** 0.25 / rho_g ** 0.5

    # Governing critical velocity (water always controls unless no water)
    v_critical = v_crit_w

    # Convert to volumetric rate: Q (Mcf/d) = v * A * 86400 / (1000 * Bg)
    # where Bg in ft3/scf
    area = math.pi / 4.0 * (tubing_id_in / 12.0) ** 2
    z = _gas_z_factor_simple(wellhead_temp_f, wellhead_pressure_psi, gas_sg)
    temp_r = wellhead_temp_f + 460.0
    # q_crit (Mcf/d) = v * A * 86400 * P / (z * T_R * 10.73) / 1000
    # Rearranged from gas law
    q_crit_water = v_crit_w * area * 86400.0 * wellhead_pressure_psi / (z * temp_r * 10.73 * 1000.0)

    q_crit_condensate = None
    if v_crit_c is not None:
        q_crit_condensate = v_crit_c * area * 86400.0 * wellhead_pressure_psi / (z * temp_r * 10.73 * 1000.0)

    q_critical = q_crit_water

    result = {
        "inputs": {
            "wellhead_pressure_psi": wellhead_pressure_psi,
            "wellhead_temp_f": wellhead_temp_f,
            "gas_sg": gas_sg,
            "water_sg": water_sg,
            "tubing_id_in": tubing_id_in,
        },
        "correlation": "Turner et al. (1969)",
        "units": {
            "velocity": "ft/s",
            "rate": "Mcf/d",
            "density": "lb/ft3",
        },
        "gas_density_lb_ft3": round(rho_g, 4),
        "critical_velocity_water_ft_s": round(v_crit_w, 2),
        "critical_rate_water_mcfd": round(q_crit_water, 1),
    }

    if condensate_sg is not None:
        result["inputs"]["condensate_sg"] = condensate_sg
        result["critical_velocity_condensate_ft_s"] = round(v_crit_c, 2)
        result["critical_rate_condensate_mcfd"] = round(q_crit_condensate, 1)

    if current_rate_mcfd is not None:
        result["inputs"]["current_rate_mcfd"] = current_rate_mcfd
        result["current_rate_mcfd"] = current_rate_mcfd
        if current_rate_mcfd >= q_critical:
            result["status"] = "unloading (above critical rate)"
        else:
            pct_below = round((1.0 - current_rate_mcfd / q_critical) * 100, 1)
            result["status"] = f"loading (below critical rate by {pct_below}%)"

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# 3. Coleman Critical Rate
# ---------------------------------------------------------------------------

def calculate_coleman_critical_rate(
    wellhead_pressure_psi: float,
    wellhead_temp_f: float,
    gas_sg: float,
    tubing_id_in: float = 2.441,
    current_rate_mcfd: float | None = None,
) -> str:
    """Coleman et al. (1991) critical rate for liquid loading.

    Applies a 20% reduction to Turner critical velocity for low-pressure
    gas wells (< ~500 psi wellhead pressure).

    Args:
        wellhead_pressure_psi: Wellhead flowing pressure in psi.
        wellhead_temp_f: Wellhead temperature in degrees F.
        gas_sg: Gas specific gravity (air = 1.0).
        tubing_id_in: Tubing inner diameter in inches. Default 2.441.
        current_rate_mcfd: Current gas rate in Mcf/d for status check (optional).

    Returns:
        JSON string with critical velocity, critical rate, and loading status.
    """
    _validate_positive("wellhead_pressure_psi", wellhead_pressure_psi)
    _validate_positive("wellhead_temp_f", wellhead_temp_f)
    _validate_positive("gas_sg", gas_sg)
    _validate_positive("tubing_id_in", tubing_id_in)

    # Gas density
    rho_g = _gas_density(wellhead_pressure_psi, wellhead_temp_f, gas_sg)

    # Water properties
    rho_w = 1.07 * 62.4
    sigma_w = _surface_tension_water()

    # Coleman: 20% reduction from Turner
    # v_crit = 1.593 * 0.8 * ... = effectively using coefficient ~1.274
    # Actually Coleman just multiplies Turner velocity by 0.8
    v_turner = 1.593 * sigma_w ** 0.25 * (rho_w - rho_g) ** 0.25 / rho_g ** 0.5
    v_critical = v_turner * 0.8  # 20% reduction

    # Volumetric rate
    area = math.pi / 4.0 * (tubing_id_in / 12.0) ** 2
    z = _gas_z_factor_simple(wellhead_temp_f, wellhead_pressure_psi, gas_sg)
    temp_r = wellhead_temp_f + 460.0
    q_critical = v_critical * area * 86400.0 * wellhead_pressure_psi / (z * temp_r * 10.73 * 1000.0)

    result = {
        "inputs": {
            "wellhead_pressure_psi": wellhead_pressure_psi,
            "wellhead_temp_f": wellhead_temp_f,
            "gas_sg": gas_sg,
            "tubing_id_in": tubing_id_in,
        },
        "correlation": "Coleman et al. (1991)",
        "units": {
            "velocity": "ft/s",
            "rate": "Mcf/d",
            "density": "lb/ft3",
        },
        "gas_density_lb_ft3": round(rho_g, 4),
        "turner_velocity_ft_s": round(v_turner, 2),
        "critical_velocity_ft_s": round(v_critical, 2),
        "critical_rate_mcfd": round(q_critical, 1),
        "note": "Coleman applies 20% reduction to Turner for low-pressure wells",
    }

    if current_rate_mcfd is not None:
        result["inputs"]["current_rate_mcfd"] = current_rate_mcfd
        result["current_rate_mcfd"] = current_rate_mcfd
        if current_rate_mcfd >= q_critical:
            result["status"] = "unloading (above critical rate)"
        else:
            pct_below = round((1.0 - current_rate_mcfd / q_critical) * 100, 1)
            result["status"] = f"loading (below critical rate by {pct_below}%)"

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# 4. Hydrate Temperature (Katz correlation)
# ---------------------------------------------------------------------------

def calculate_hydrate_temperature(
    pressure_psi: float,
    gas_sg: float,
) -> str:
    """Estimate hydrate formation temperature using gas-gravity method (Katz).

    Empirical correlation equivalent to the Katz gas-gravity chart for
    predicting gas hydrate formation conditions.

    Args:
        pressure_psi: System pressure in psi.
        gas_sg: Gas specific gravity (air = 1.0).

    Returns:
        JSON string with hydrate formation temperature.
    """
    _validate_positive("pressure_psi", pressure_psi)
    _validate_positive("gas_sg", gas_sg)
    if gas_sg < 0.55 or gas_sg > 1.0:
        raise ValueError(f"gas_sg should be between 0.55 and 1.0 for this correlation, got {gas_sg}")

    # Katz gas-gravity correlation (polynomial fit to chart)
    # T_hydrate (F) = a * ln(P) + b
    # Adjusted by gas gravity
    ln_p = math.log(pressure_psi)

    # Base correlation for 0.6 SG gas (fitted to Katz chart data)
    t_base = -83.84 + 34.66 * ln_p - 1.776 * ln_p ** 2

    # Gas gravity correction (heavier gas = higher hydrate temperature)
    sg_correction = (gas_sg - 0.6) * 40.0

    t_hydrate = t_base + sg_correction

    result = {
        "inputs": {
            "pressure_psi": pressure_psi,
            "gas_sg": gas_sg,
        },
        "correlation": "Katz gas-gravity chart (empirical fit)",
        "units": {
            "temperature": "deg_F",
            "pressure": "psi",
        },
        "hydrate_temperature_f": round(t_hydrate, 1),
        "note": "Operate above this temperature to avoid hydrate formation",
    }

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# 5. Hydrate Inhibitor Dosing (Hammerschmidt)
# ---------------------------------------------------------------------------

_INHIBITOR_PROPS = {
    "methanol": {"M": 32.04, "K": 2335.0, "density_lb_gal": 6.63},
    "meg": {"M": 62.07, "K": 2700.0, "density_lb_gal": 9.35},
    "ethanol": {"M": 46.07, "K": 2335.0, "density_lb_gal": 6.58},
}


def calculate_hydrate_inhibitor_dosing(
    hydrate_temp_f: float,
    operating_temp_f: float,
    water_rate_bwpd: float,
    inhibitor: str = "methanol",
) -> str:
    """Calculate hydrate inhibitor injection rate using Hammerschmidt equation.

    dT = K * W / (M * (100 - W))

    Solves for W (weight percent of inhibitor in water phase), then computes
    the required volume injection rate.

    Args:
        hydrate_temp_f: Hydrate formation temperature in degrees F.
        operating_temp_f: Target operating temperature in degrees F.
        water_rate_bwpd: Water production rate in bbl/day.
        inhibitor: Inhibitor type - 'methanol', 'meg', or 'ethanol'.

    Returns:
        JSON string with required weight percent and injection rate.
    """
    inhibitor = inhibitor.lower().strip()
    if inhibitor not in _INHIBITOR_PROPS:
        raise ValueError(f"Unsupported inhibitor '{inhibitor}'. Use: {list(_INHIBITOR_PROPS.keys())}")

    _validate_positive("water_rate_bwpd", water_rate_bwpd)

    # Temperature depression needed
    dt = hydrate_temp_f - operating_temp_f

    if dt <= 0:
        return json.dumps({
            "inputs": {
                "hydrate_temp_f": hydrate_temp_f,
                "operating_temp_f": operating_temp_f,
                "water_rate_bwpd": water_rate_bwpd,
                "inhibitor": inhibitor,
            },
            "correlation": "Hammerschmidt (1934)",
            "note": "Operating temperature is already below hydrate temperature. No inhibitor needed.",
            "inhibitor_weight_pct": 0.0,
            "inhibitor_rate_gal_day": 0.0,
        }, indent=2)

    props = _INHIBITOR_PROPS[inhibitor]
    mw = props["M"]
    k = props["K"]
    density = props["density_lb_gal"]

    # Hammerschmidt: dT = K * W / (M * (100 - W))
    # Solve for W: W = 100 * M * dT / (K + M * dT)
    w = 100.0 * mw * dt / (k + mw * dt)
    w = min(w, 95.0)  # Cap at reasonable max

    # Mass rate of inhibitor needed
    # Water mass rate: water_rate_bwpd * 42 gal/bbl * 8.34 lb/gal (approx for brine)
    water_mass_lb_day = water_rate_bwpd * 42.0 * 8.34 * 1.07  # account for brine SG
    # w = mass_inhibitor / (mass_inhibitor + mass_water) * 100
    # mass_inhibitor = w * mass_water / (100 - w)
    inhibitor_mass_lb_day = w * water_mass_lb_day / (100.0 - w)
    inhibitor_rate_gal_day = inhibitor_mass_lb_day / density

    result = {
        "inputs": {
            "hydrate_temp_f": hydrate_temp_f,
            "operating_temp_f": operating_temp_f,
            "water_rate_bwpd": water_rate_bwpd,
            "inhibitor": inhibitor,
        },
        "correlation": "Hammerschmidt (1934)",
        "units": {
            "temperature": "deg_F",
            "rate": "gal/day",
            "weight_percent": "%",
        },
        "temperature_depression_f": round(dt, 1),
        "inhibitor_molecular_weight": mw,
        "inhibitor_weight_pct": round(w, 2),
        "inhibitor_rate_lb_day": round(inhibitor_mass_lb_day, 1),
        "inhibitor_rate_gal_day": round(inhibitor_rate_gal_day, 1),
    }

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# 6. Erosional Velocity (API RP 14E)
# ---------------------------------------------------------------------------

def calculate_erosional_velocity(
    density_mix_lb_ft3: float,
    c_factor: float = 100.0,
) -> str:
    """Calculate erosional velocity per API RP 14E.

    v_e = C / sqrt(rho_mix)

    Used to size production piping and check for erosion risk.

    Args:
        density_mix_lb_ft3: Mixture density in lb/ft3.
        c_factor: Erosional constant (dimensionless). Default 100.
            Common values: 100 (continuous service), 125 (intermittent),
            150-200 (inhibited/corrosion-resistant alloys).

    Returns:
        JSON string with erosional velocity.
    """
    _validate_positive("density_mix_lb_ft3", density_mix_lb_ft3)
    _validate_positive("c_factor", c_factor)

    v_erosional = c_factor / math.sqrt(density_mix_lb_ft3)

    result = {
        "inputs": {
            "density_mix_lb_ft3": density_mix_lb_ft3,
            "c_factor": c_factor,
        },
        "correlation": "API RP 14E",
        "units": {
            "velocity": "ft/s",
            "density": "lb/ft3",
        },
        "erosional_velocity_ft_s": round(v_erosional, 2),
        "note": "Maintain actual velocity below erosional velocity to prevent pipe damage",
    }

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# 7. Critical Choke Flow (Gilbert Correlation)
# ---------------------------------------------------------------------------

def calculate_critical_choke_flow(
    upstream_pressure_psi: float,
    choke_size_64ths: float,
    gor_scf_bbl: float,
    oil_api: float,
    water_cut: float = 0.0,
    gas_sg: float = 0.65,
) -> str:
    """Calculate flow rate through a choke using Gilbert correlation (1954).

    q = P_upstream * S^1.89 / (435 * GLR^0.546)

    Applicable for critical (sonic) flow through wellhead chokes.

    Args:
        upstream_pressure_psi: Upstream (casing/tubing) pressure in psi.
        choke_size_64ths: Choke bean size in 64ths of an inch.
        gor_scf_bbl: Gas-oil ratio in scf/bbl.
        oil_api: Oil API gravity.
        water_cut: Water cut as fraction (0-1). Default 0.0.
        gas_sg: Gas specific gravity (air = 1.0). Default 0.65.

    Returns:
        JSON string with estimated flow rate through choke.
    """
    _validate_positive("upstream_pressure_psi", upstream_pressure_psi)
    _validate_positive("choke_size_64ths", choke_size_64ths)
    _validate_positive("gor_scf_bbl", gor_scf_bbl)
    _validate_positive("oil_api", oil_api)
    _validate_non_negative("water_cut", water_cut)
    if water_cut > 1.0:
        raise ValueError(f"water_cut must be between 0 and 1, got {water_cut}")

    # GLR (gas-liquid ratio in scf/bbl total liquid)
    # If water cut > 0, GOR is per oil bbl, so GLR = GOR * (1 - WC)
    glr = gor_scf_bbl * (1.0 - water_cut)
    if glr <= 0:
        raise ValueError("Effective GLR must be positive for Gilbert correlation")

    # Gilbert correlation
    # q (bbl/d total liquid) = P * S^1.89 / (435 * GLR^0.546)
    q_total_liquid = upstream_pressure_psi * choke_size_64ths ** 1.89 / (435.0 * glr ** 0.546)
    q_oil = q_total_liquid * (1.0 - water_cut)
    q_water = q_total_liquid * water_cut
    q_gas_mcfd = q_oil * gor_scf_bbl / 1000.0

    result = {
        "inputs": {
            "upstream_pressure_psi": upstream_pressure_psi,
            "choke_size_64ths": choke_size_64ths,
            "gor_scf_bbl": gor_scf_bbl,
            "oil_api": oil_api,
            "water_cut": water_cut,
            "gas_sg": gas_sg,
        },
        "correlation": "Gilbert (1954)",
        "units": {
            "liquid_rate": "bbl/d",
            "gas_rate": "Mcf/d",
            "glr": "scf/bbl",
        },
        "effective_glr_scf_bbl": round(glr, 1),
        "total_liquid_rate_bpd": round(q_total_liquid, 1),
        "oil_rate_bopd": round(q_oil, 1),
        "water_rate_bwpd": round(q_water, 1),
        "gas_rate_mcfd": round(q_gas_mcfd, 1),
        "choke_size_inches": round(choke_size_64ths / 64.0, 4),
        "note": "Valid for critical (sonic) flow only. Verify P_downstream / P_upstream < 0.55",
    }

    return json.dumps(result, indent=2)
