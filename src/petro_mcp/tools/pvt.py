"""PVT (Pressure-Volume-Temperature) black-oil fluid property correlations.

Implements the most commonly used black-oil correlations in petroleum
engineering for estimating fluid properties from readily available field data.

Correlations used:
    - Standing (1947): Bubble point pressure, solution GOR, oil FVF
    - Beggs and Robinson (1975): Dead and live oil viscosity
    - Sutton (1985): Gas pseudocritical properties
    - Hall and Yarborough (1973): Gas Z-factor
    - Lee, Gonzalez, and Eakin (1966): Gas viscosity

References:
    Standing, M.B., "A Pressure-Volume-Temperature Correlation for Mixtures
        of California Oils and Gases," API Drilling and Production Practice, 1947.
    Beggs, H.D. and Robinson, J.R., "Estimating the Viscosity of Crude Oil
        Systems," JPT, September 1975, pp. 1140-1141.
    Sutton, R.P., "Compressibility Factors for High-Molecular-Weight Reservoir
        Gases," SPE 14265, 1985.
    Hall, K.R. and Yarborough, L., "A New Equation of State for Z-Factor
        Calculations," Oil and Gas Journal, June 1973.
    Lee, A.L., Gonzalez, M.H., and Eakin, B.E., "The Viscosity of Natural
        Gases," JPT, August 1966, pp. 997-1000.
"""

from __future__ import annotations

import json
import math


def _validate_pvt_inputs(
    api_gravity: float,
    gas_sg: float,
    temperature: float,
    pressure: float,
) -> None:
    """Validate common PVT input parameters."""
    if api_gravity <= 0:
        raise ValueError("API gravity must be positive")
    if gas_sg <= 0:
        raise ValueError("Gas specific gravity must be positive")
    if temperature <= 0:
        raise ValueError("Temperature must be positive (°F)")
    if pressure <= 0:
        raise ValueError("Pressure must be positive (psi)")


def _standing_rs(
    pressure: float,
    temperature: float,
    api_gravity: float,
    gas_sg: float,
) -> float:
    """Solution gas-oil ratio using Standing's correlation (1947).

    Rs = Sg * ((P / 18.2 + 1.4) * 10^(0.0125*API - 0.00091*T))^1.2048

    Args:
        pressure: Pressure in psi.
        temperature: Temperature in °F.
        api_gravity: Oil API gravity.
        gas_sg: Gas specific gravity (air = 1.0).

    Returns:
        Solution GOR in scf/STB.
    """
    exponent = 0.0125 * api_gravity - 0.00091 * temperature
    yg = gas_sg * ((pressure / 18.2 + 1.4) * 10**exponent) ** 1.2048
    return max(yg, 0.0)


def _standing_pb(
    rs: float,
    temperature: float,
    api_gravity: float,
    gas_sg: float,
) -> float:
    """Bubble point pressure using Standing's correlation (1947).

    Pb = 18.2 * (Rs^0.83 * 10^(0.00091*T - 0.0125*API) - 1.4)

    Args:
        rs: Solution gas-oil ratio in scf/STB.
        temperature: Temperature in °F.
        api_gravity: Oil API gravity.
        gas_sg: Gas specific gravity (air = 1.0).

    Returns:
        Bubble point pressure in psi.
    """
    if rs <= 0:
        return 14.7  # atmospheric
    exponent = 0.00091 * temperature - 0.0125 * api_gravity
    pb = 18.2 * ((rs / gas_sg) ** 0.83 * 10**exponent - 1.4)
    return max(pb, 14.7)


def _standing_bo(
    rs: float,
    temperature: float,
    api_gravity: float,
    gas_sg: float,
) -> float:
    """Oil formation volume factor using Standing's correlation (1947).

    Bo = 0.9759 + 0.000120 * (Rs * (Sg/So)^0.5 + 1.25*T)^1.2

    Args:
        rs: Solution gas-oil ratio in scf/STB.
        temperature: Temperature in °F.
        api_gravity: Oil API gravity.
        gas_sg: Gas specific gravity (air = 1.0).

    Returns:
        Oil FVF in bbl/STB.
    """
    # Oil specific gravity from API gravity
    oil_sg = 141.5 / (api_gravity + 131.5)
    f_factor = rs * (gas_sg / oil_sg) ** 0.5 + 1.25 * temperature
    bo = 0.9759 + 0.000120 * f_factor**1.2
    return bo


def _oil_density(
    rs: float,
    temperature: float,
    api_gravity: float,
    gas_sg: float,
    bo: float,
) -> float:
    """Oil density at reservoir conditions.

    rho_o = (350 * gamma_o + 0.0764 * Rs * gamma_g) / (5.615 * Bo)

    Args:
        rs: Solution GOR in scf/STB.
        temperature: Temperature in °F.
        api_gravity: Oil API gravity.
        gas_sg: Gas specific gravity.
        bo: Oil FVF in bbl/STB.

    Returns:
        Oil density in lb/ft³.
    """
    oil_sg = 141.5 / (api_gravity + 131.5)
    # 350 lb/bbl for water at 60°F, 0.0764 lb/scf for air at standard conditions
    rho = (350.0 * oil_sg + 0.0764 * rs * gas_sg) / (5.615 * bo)
    return rho


def _beggs_robinson_dead_oil_viscosity(
    temperature: float,
    api_gravity: float,
) -> float:
    """Dead oil viscosity using Beggs and Robinson correlation (1975).

    mu_od = 10^(10^(3.0324 - 0.02023*API) * T^(-1.163)) - 1

    Args:
        temperature: Temperature in °F.
        api_gravity: Oil API gravity.

    Returns:
        Dead oil viscosity in cp.
    """
    x = 10 ** (3.0324 - 0.02023 * api_gravity) * temperature ** (-1.163)
    mu_od = 10**x - 1.0
    return mu_od


def _beggs_robinson_live_oil_viscosity(
    mu_od: float,
    rs: float,
) -> float:
    """Live (saturated) oil viscosity using Beggs and Robinson correlation (1975).

    A = 10.715 * (Rs + 100)^(-0.515)
    B = 5.44 * (Rs + 150)^(-0.338)
    mu_o = A * mu_od^B

    Args:
        mu_od: Dead oil viscosity in cp.
        rs: Solution GOR in scf/STB.

    Returns:
        Live oil viscosity in cp.
    """
    a = 10.715 * (rs + 100.0) ** (-0.515)
    b = 5.44 * (rs + 150.0) ** (-0.338)
    mu_o = a * mu_od**b
    return mu_o


def _sutton_pseudocritical(gas_sg: float) -> tuple[float, float]:
    """Gas pseudocritical properties using Sutton's correlation (1985).

    Tpc = 169.2 + 349.5 * gamma_g - 74.0 * gamma_g^2  (°R)
    Ppc = 756.8 - 131.0 * gamma_g - 3.6 * gamma_g^2   (psia)

    Args:
        gas_sg: Gas specific gravity (air = 1.0).

    Returns:
        Tuple of (Tpc in °R, Ppc in psia).
    """
    tpc = 169.2 + 349.5 * gas_sg - 74.0 * gas_sg**2
    ppc = 756.8 - 131.0 * gas_sg - 3.6 * gas_sg**2
    return tpc, ppc


def _hall_yarborough_z(
    temperature: float,
    pressure: float,
    gas_sg: float,
) -> float:
    """Gas Z-factor using the Hall-Yarborough method (1973).

    Uses Newton-Raphson iteration to solve the Hall-Yarborough equation
    of state for the reduced density, then calculates Z.

    Args:
        temperature: Temperature in °F.
        pressure: Pressure in psi.
        gas_sg: Gas specific gravity.

    Returns:
        Gas compressibility factor (Z).
    """
    tpc, ppc = _sutton_pseudocritical(gas_sg)
    t_rankine = temperature + 459.67
    t_pr = t_rankine / tpc
    p_pr = pressure / ppc

    # Reciprocal of reduced temperature
    t_inv = 1.0 / t_pr

    # Hall-Yarborough constants
    a1 = -0.06125 * p_pr * t_inv * math.exp(-1.2 * (1.0 - t_inv) ** 2)
    a2 = 14.76 * t_inv - 9.76 * t_inv**2 + 4.58 * t_inv**3
    a3 = 90.7 * t_inv - 242.2 * t_inv**2 + 42.4 * t_inv**3
    a4 = 2.18 + 2.82 * t_inv

    # Newton-Raphson iteration for reduced density y
    y = 0.001  # initial guess
    for _ in range(100):
        fy = (
            a1
            + (y + y**2 + y**3 - y**4) / (1.0 - y) ** 3
            - a2 * y**2
            + a3 * y**a4
        )
        # Derivative
        dfy = (
            (1.0 + 4.0 * y + 4.0 * y**2 - 4.0 * y**3 + y**4)
            / (1.0 - y) ** 4
            - 2.0 * a2 * y
            + a3 * a4 * y ** (a4 - 1.0)
        )
        if abs(dfy) < 1e-30:
            break
        y_new = y - fy / dfy
        # Keep y in physical bounds
        y_new = max(y_new, 1e-10)
        y_new = min(y_new, 0.9999)
        if abs(y_new - y) < 1e-12:
            y = y_new
            break
        y = y_new

    # Z-factor from reduced density
    z = -a1 / y if y > 1e-15 else 1.0
    # Clamp to physical range
    z = max(z, 0.05)
    return z


def _gas_fvf(z: float, temperature: float, pressure: float) -> float:
    """Gas formation volume factor.

    Bg = 0.02829 * Z * T_rankine / P  (rcf/scf)

    Args:
        z: Gas Z-factor.
        temperature: Temperature in °F.
        pressure: Pressure in psi.

    Returns:
        Gas FVF in rcf/scf.
    """
    t_rankine = temperature + 459.67
    bg = 0.02829 * z * t_rankine / pressure
    return bg


def _lee_gonzalez_eakin_viscosity(
    temperature: float,
    pressure: float,
    z: float,
    gas_sg: float,
) -> float:
    """Gas viscosity using Lee, Gonzalez, and Eakin correlation (1966).

    mu_g = K * exp(X * rho_g^Y) * 1e-4

    Args:
        temperature: Temperature in °F.
        pressure: Pressure in psi.
        z: Gas Z-factor.
        gas_sg: Gas specific gravity.

    Returns:
        Gas viscosity in cp.
    """
    t_rankine = temperature + 459.67
    # Molecular weight of gas
    mg = 28.97 * gas_sg

    # Gas density in g/cm³
    rho_g = pressure * mg / (z * t_rankine * 10.7316)  # lb/ft³
    rho_g_gcc = rho_g / 62.428  # convert to g/cm³

    k = (9.4 + 0.02 * mg) * t_rankine**1.5 / (209.0 + 19.0 * mg + t_rankine)
    x = 3.5 + 986.0 / t_rankine + 0.01 * mg
    y = 2.4 - 0.2 * x

    mu_g = k * math.exp(x * rho_g_gcc**y) * 1e-4
    return mu_g


def _gas_compressibility(
    z: float,
    pressure: float,
    gas_sg: float,
    temperature: float,
    dp_frac: float = 0.001,
) -> float:
    """Gas compressibility (isothermal) from numerical differentiation of Z.

    cg = 1/P - 1/Z * (dZ/dP)_T

    Uses central difference for dZ/dP.

    Args:
        z: Z-factor at current pressure.
        pressure: Current pressure in psi.
        gas_sg: Gas specific gravity.
        temperature: Temperature in °F.
        dp_frac: Fractional pressure step for numerical derivative.

    Returns:
        Gas compressibility in 1/psi.
    """
    dp = pressure * dp_frac
    z_plus = _hall_yarborough_z(temperature, pressure + dp, gas_sg)
    z_minus = _hall_yarborough_z(temperature, pressure - dp, gas_sg)
    dz_dp = (z_plus - z_minus) / (2.0 * dp)
    cg = 1.0 / pressure - (1.0 / z) * dz_dp
    return cg


def bubble_point(
    api_gravity: float,
    gas_sg: float,
    temperature: float,
    rs: float,
) -> str:
    """Calculate bubble point pressure using Standing's correlation (1947).

    Pb = 18.2 * ((Rs/Sg)^0.83 * 10^(0.00091*T - 0.0125*API) - 1.4)

    Args:
        api_gravity: Oil API gravity (degrees).
        gas_sg: Gas specific gravity (air = 1.0).
        temperature: Reservoir temperature in °F.
        rs: Solution gas-oil ratio at bubble point in scf/STB.

    Returns:
        JSON string with bubble point pressure and inputs.
    """
    if api_gravity <= 0:
        raise ValueError("API gravity must be positive")
    if gas_sg <= 0:
        raise ValueError("Gas specific gravity must be positive")
    if temperature <= 0:
        raise ValueError("Temperature must be positive (°F)")
    if rs < 0:
        raise ValueError("Solution GOR must be non-negative")

    pb = _standing_pb(rs, temperature, api_gravity, gas_sg)

    result = {
        "bubble_point_pressure_psi": round(pb, 1),
        "correlation": "Standing (1947)",
        "inputs": {
            "api_gravity": api_gravity,
            "gas_sg": gas_sg,
            "temperature_F": temperature,
            "solution_gor_scf_stb": rs,
        },
        "units": {"bubble_point_pressure": "psi"},
    }
    return json.dumps(result, indent=2)


def calculate_pvt(
    api_gravity: float,
    gas_sg: float,
    temperature: float,
    pressure: float,
    separator_pressure: float = 100.0,
    separator_temperature: float = 60.0,
) -> str:
    """Calculate comprehensive black-oil PVT properties at given conditions.

    Computes oil, gas, and fluid properties using published correlations:
        - Standing (1947): Pb, Rs, Bo
        - Beggs and Robinson (1975): Oil viscosity
        - Sutton (1985): Gas pseudocritical properties
        - Hall and Yarborough (1973): Gas Z-factor
        - Lee, Gonzalez, and Eakin (1966): Gas viscosity

    Args:
        api_gravity: Oil API gravity (degrees).
        gas_sg: Gas specific gravity (air = 1.0).
        temperature: Reservoir temperature in °F.
        pressure: Current reservoir pressure in psi.
        separator_pressure: Separator pressure in psi (default 100).
        separator_temperature: Separator temperature in °F (default 60).

    Returns:
        JSON string with all calculated PVT properties, units, and
        correlations used.
    """
    _validate_pvt_inputs(api_gravity, gas_sg, temperature, pressure)

    # --- Oil properties ---
    # Solution GOR at current pressure (Standing)
    rs = _standing_rs(pressure, temperature, api_gravity, gas_sg)

    # Bubble point pressure (Standing) — using Rs at current P
    pb = _standing_pb(rs, temperature, api_gravity, gas_sg)

    # If pressure > Pb, oil is undersaturated: Rs is fixed at Rs(Pb)
    # and the actual Pb is as calculated. If P < Pb, Rs varies with P.
    if pressure >= pb:
        # Undersaturated — Rs is at its maximum (at Pb)
        rs_at_pb = rs
    else:
        # Saturated — Rs is at current pressure
        rs_at_pb = _standing_rs(pb, temperature, api_gravity, gas_sg)

    # Oil FVF (Standing)
    bo = _standing_bo(rs, temperature, api_gravity, gas_sg)

    # Oil density at reservoir conditions
    rho_o = _oil_density(rs, temperature, api_gravity, gas_sg, bo)

    # Dead oil viscosity (Beggs-Robinson)
    mu_od = _beggs_robinson_dead_oil_viscosity(temperature, api_gravity)

    # Live oil viscosity (Beggs-Robinson)
    mu_o = _beggs_robinson_live_oil_viscosity(mu_od, rs)

    # --- Gas properties ---
    # Pseudocritical properties (Sutton)
    tpc, ppc = _sutton_pseudocritical(gas_sg)

    # Z-factor (Hall-Yarborough)
    z = _hall_yarborough_z(temperature, pressure, gas_sg)

    # Gas FVF
    bg = _gas_fvf(z, temperature, pressure)

    # Gas viscosity (Lee-Gonzalez-Eakin)
    mu_g = _lee_gonzalez_eakin_viscosity(temperature, pressure, z, gas_sg)

    # Gas compressibility
    cg = _gas_compressibility(z, pressure, gas_sg, temperature)

    # Gas density at reservoir conditions (lb/ft³)
    mg = 28.97 * gas_sg
    t_rankine = temperature + 459.67
    rho_g = pressure * mg / (z * t_rankine * 10.7316)

    result = {
        "inputs": {
            "api_gravity": api_gravity,
            "gas_sg": gas_sg,
            "temperature_F": temperature,
            "pressure_psi": pressure,
            "separator_pressure_psi": separator_pressure,
            "separator_temperature_F": separator_temperature,
        },
        "oil_properties": {
            "bubble_point_pressure_psi": round(pb, 1),
            "bubble_point_correlation": "Standing (1947)",
            "solution_gor_scf_stb": round(rs, 1),
            "solution_gor_correlation": "Standing (1947)",
            "oil_fvf_bbl_stb": round(bo, 4),
            "oil_fvf_correlation": "Standing (1947)",
            "oil_density_lb_ft3": round(rho_o, 2),
            "dead_oil_viscosity_cp": round(mu_od, 4),
            "live_oil_viscosity_cp": round(mu_o, 4),
            "oil_viscosity_correlation": "Beggs and Robinson (1975)",
        },
        "gas_properties": {
            "z_factor": round(z, 4),
            "z_factor_correlation": "Hall and Yarborough (1973)",
            "gas_fvf_rcf_scf": round(bg, 6),
            "gas_density_lb_ft3": round(rho_g, 4),
            "gas_viscosity_cp": round(mu_g, 6),
            "gas_viscosity_correlation": "Lee, Gonzalez, and Eakin (1966)",
            "gas_compressibility_1_psi": round(cg, 8),
            "pseudocritical_temperature_R": round(tpc, 1),
            "pseudocritical_pressure_psia": round(ppc, 1),
            "pseudocritical_correlation": "Sutton (1985)",
        },
        "units": {
            "bubble_point_pressure": "psi",
            "solution_gor": "scf/STB",
            "oil_fvf": "bbl/STB",
            "oil_density": "lb/ft³",
            "oil_viscosity": "cp",
            "z_factor": "dimensionless",
            "gas_fvf": "rcf/scf",
            "gas_density": "lb/ft³",
            "gas_viscosity": "cp",
            "gas_compressibility": "1/psi",
        },
    }

    return json.dumps(result, indent=2)
