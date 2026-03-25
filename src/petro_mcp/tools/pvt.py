"""PVT (Pressure-Volume-Temperature) black-oil fluid property correlations.

Implements the most commonly used black-oil correlations in petroleum
engineering for estimating fluid properties from readily available field data.

Correlations used:
    - Standing (1947): Bubble point pressure, solution GOR, oil FVF
    - Vasquez and Beggs (1980): Solution GOR, oil FVF, oil compressibility
    - Al-Marhoun (1988): Bubble point pressure
    - Petrosky and Farshad (1993): Bubble point, solution GOR, oil FVF
    - Beggs and Robinson (1975): Dead and live oil viscosity
    - Sutton (1985): Gas pseudocritical properties
    - Piper, McCain, and Corredor (1993): Gas pseudocritical properties
    - Hall and Yarborough (1973): Gas Z-factor
    - Dranchuk and Abou-Kassem (1975): Gas Z-factor
    - Lee, Gonzalez, and Eakin (1966): Gas viscosity
    - McCain (1990): Brine density, viscosity, FVF
    - Osif (1988): Brine compressibility

References:
    Standing, M.B., "A Pressure-Volume-Temperature Correlation for Mixtures
        of California Oils and Gases," API Drilling and Production Practice, 1947.
    Vasquez, M.E. and Beggs, H.D., "Correlations for Fluid Physical Property
        Prediction," JPT, June 1980, pp. 968-970.
    Al-Marhoun, M.A., "PVT Correlations for Middle East Crude Oils," JPT,
        May 1988, pp. 650-666.
    Petrosky, G.E. and Farshad, F.F., "Pressure-Volume-Temperature
        Correlations for Gulf of Mexico Crude Oils," SPE 26644, 1993.
    Beggs, H.D. and Robinson, J.R., "Estimating the Viscosity of Crude Oil
        Systems," JPT, September 1975, pp. 1140-1141.
    Sutton, R.P., "Compressibility Factors for High-Molecular-Weight Reservoir
        Gases," SPE 14265, 1985.
    Piper, L.D., McCain, W.D., and Corredor, J.H., "Compressibility Factors
        for Naturally Occurring Petroleum Gases," SPE 26668, 1993.
    Hall, K.R. and Yarborough, L., "A New Equation of State for Z-Factor
        Calculations," Oil and Gas Journal, June 1973.
    Dranchuk, P.M. and Abou-Kassem, J.H., "Calculation of Z Factors for
        Natural Gases Using Equations of State," JCPT, July-September 1975.
    Lee, A.L., Gonzalez, M.H., and Eakin, B.E., "The Viscosity of Natural
        Gases," JPT, August 1966, pp. 997-1000.
    McCain, W.D., "The Properties of Petroleum Fluids," PennWell, 1990.
    Osif, T.L., "The Effects of Salt, Gas, Temperature, and Pressure on the
        Compressibility of Water," SPE Reservoir Engineering, February 1988.
"""

from __future__ import annotations

import json
import math


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

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


def _validate_brine_inputs(
    temperature: float,
    pressure: float,
    salinity: float,
) -> None:
    """Validate brine property input parameters."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive (°F)")
    if pressure <= 0:
        raise ValueError("Pressure must be positive (psi)")
    if salinity < 0:
        raise ValueError("Salinity must be non-negative (ppm)")
    if salinity > 300000:
        raise ValueError("Salinity must be <= 300,000 ppm")


# ---------------------------------------------------------------------------
# Oil PVT — Standing (1947)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Oil PVT — Vasquez and Beggs (1980)
# ---------------------------------------------------------------------------

def _vasquez_beggs_coefficients(api_gravity: float) -> dict:
    """Return Vasquez-Beggs coefficients based on API gravity.

    Two sets of coefficients: API <= 30 and API > 30.
    """
    if api_gravity <= 30:
        return {
            "C1": 0.0362, "C2": 1.0937, "C3": 25.724,
            "C4": 4.677e-4, "C5": 1.751e-5, "C6": -1.811e-8,
        }
    return {
        "C1": 0.0178, "C2": 1.187, "C3": 23.931,
        "C4": 4.670e-4, "C5": 1.100e-5, "C6": 1.337e-9,
    }


def _vasquez_beggs_rs(
    pressure: float,
    temperature: float,
    api_gravity: float,
    gas_sg: float,
    separator_pressure: float = 100.0,
) -> float:
    """Solution GOR using Vasquez and Beggs correlation (1980).

    Rs = C1 * Sg_corrected * P^C2 * exp(C3 * API / T_R)

    Args:
        pressure: Pressure in psi.
        temperature: Temperature in °F.
        api_gravity: Oil API gravity.
        gas_sg: Gas specific gravity (air = 1.0).
        separator_pressure: Separator pressure in psi.

    Returns:
        Solution GOR in scf/STB.
    """
    t_rankine = temperature + 459.67
    # Corrected gas SG to 100 psig separator
    sg_corr = gas_sg * (
        1.0 + 5.912e-5 * api_gravity * (temperature - 60.0)
        * math.log10(separator_pressure / 114.7)
    )
    c = _vasquez_beggs_coefficients(api_gravity)
    rs = c["C1"] * sg_corr * pressure ** c["C2"] * math.exp(
        c["C3"] * api_gravity / t_rankine
    )
    return max(rs, 0.0)


def _vasquez_beggs_bo(
    rs: float,
    temperature: float,
    api_gravity: float,
    gas_sg: float,
    separator_pressure: float = 100.0,
) -> float:
    """Oil FVF using Vasquez and Beggs correlation (1980).

    Bo = 1.0 + C4*Rs + C5*(T-60)*(API/Sg_corr) + C6*Rs*(T-60)*(API/Sg_corr)

    Args:
        rs: Solution GOR in scf/STB.
        temperature: Temperature in °F.
        api_gravity: Oil API gravity.
        gas_sg: Gas specific gravity (air = 1.0).
        separator_pressure: Separator pressure in psi.

    Returns:
        Oil FVF in bbl/STB.
    """
    sg_corr = gas_sg * (
        1.0 + 5.912e-5 * api_gravity * (temperature - 60.0)
        * math.log10(separator_pressure / 114.7)
    )
    c = _vasquez_beggs_coefficients(api_gravity)
    dt = temperature - 60.0
    ratio = api_gravity / sg_corr
    bo = 1.0 + c["C4"] * rs + c["C5"] * dt * ratio + c["C6"] * rs * dt * ratio
    return max(bo, 1.0)


def _vasquez_beggs_co_above_pb(
    temperature: float,
    api_gravity: float,
    gas_sg: float,
    pressure: float,
    rs: float,
) -> float:
    """Undersaturated oil compressibility using Vasquez and Beggs (1980).

    co = (C1 * Rs * T^C2 * API^C3) / (P^C4)
    with dedicated coefficients for compressibility.

    Args:
        temperature: Temperature in °F.
        api_gravity: Oil API gravity.
        gas_sg: Gas specific gravity.
        pressure: Pressure in psi (above Pb).
        rs: Solution GOR at bubble point in scf/STB.

    Returns:
        Oil compressibility in 1/psi.
    """
    # Vasquez-Beggs undersaturated oil compressibility
    # co = (-1433 + 5*Rs + 17.2*T - 1180*Sg + 12.61*API) / (1e5 * P)
    co = (-1433.0 + 5.0 * rs + 17.2 * temperature
          - 1180.0 * gas_sg + 12.61 * api_gravity) / (1e5 * pressure)
    return max(co, 1e-7)


def _oil_co_below_pb(
    temperature: float,
    api_gravity: float,
    gas_sg: float,
    pressure: float,
    pb: float,
    bo_at_p: float,
    bg: float,
) -> float:
    """Oil compressibility below bubble point (saturated).

    Uses the material-balance definition:
    co = -1/Bo * dBo/dP + Bg/Bo * dRs/dP

    Approximated with numerical derivatives.

    Args:
        temperature: Temperature in °F.
        api_gravity: Oil API gravity.
        gas_sg: Gas specific gravity.
        pressure: Pressure in psi (below Pb).
        pb: Bubble point pressure in psi.
        bo_at_p: Oil FVF at current pressure.
        bg: Gas FVF at current pressure.

    Returns:
        Oil compressibility in 1/psi.
    """
    dp = max(pressure * 0.001, 0.1)
    p_lo = max(pressure - dp, 1.0)
    p_hi = pressure + dp

    rs_lo = _standing_rs(p_lo, temperature, api_gravity, gas_sg)
    rs_hi = _standing_rs(p_hi, temperature, api_gravity, gas_sg)
    bo_lo = _standing_bo(rs_lo, temperature, api_gravity, gas_sg)
    bo_hi = _standing_bo(rs_hi, temperature, api_gravity, gas_sg)

    dbo_dp = (bo_hi - bo_lo) / (p_hi - p_lo)
    drs_dp = (rs_hi - rs_lo) / (p_hi - p_lo)

    co = -dbo_dp / bo_at_p + bg * drs_dp / bo_at_p
    return max(co, 1e-7)


# ---------------------------------------------------------------------------
# Oil PVT — Al-Marhoun (1988)
# ---------------------------------------------------------------------------

def _al_marhoun_pb(
    rs: float,
    temperature: float,
    api_gravity: float,
    gas_sg: float,
) -> float:
    """Bubble point pressure using Al-Marhoun correlation (1988).

    Pb = a * Rs^b * gamma_g^c * gamma_o^d * T_R^e

    Developed for Middle East crude oils.

    Args:
        rs: Solution GOR in scf/STB.
        temperature: Temperature in °F.
        api_gravity: Oil API gravity.
        gas_sg: Gas specific gravity.

    Returns:
        Bubble point pressure in psi.
    """
    if rs <= 0:
        return 14.7
    oil_sg = 141.5 / (api_gravity + 131.5)
    t_rankine = temperature + 459.67

    # Al-Marhoun coefficients
    a = 5.38088e-3
    b = 0.715082
    c = -1.87784
    d = 3.1437
    e = 1.32657

    pb = a * rs**b * gas_sg**c * oil_sg**d * t_rankine**e
    return max(pb, 14.7)


# ---------------------------------------------------------------------------
# Oil PVT — Petrosky and Farshad (1993)
# ---------------------------------------------------------------------------

def _petrosky_farshad_rs(
    pressure: float,
    temperature: float,
    api_gravity: float,
    gas_sg: float,
) -> float:
    """Solution GOR using Petrosky and Farshad correlation (1993).

    Developed for Gulf of Mexico crude oils.

    Args:
        pressure: Pressure in psi.
        temperature: Temperature in °F.
        api_gravity: Oil API gravity.
        gas_sg: Gas specific gravity.

    Returns:
        Solution GOR in scf/STB.
    """
    x = (
        7.916e-4 * api_gravity**1.5410
        - 4.561e-5 * temperature**1.3911
    )
    rs = (
        (pressure / 112.727 + 12.340)
        * gas_sg**0.8439
        * 10**x
    ) ** 1.73184
    return max(rs, 0.0)


def _petrosky_farshad_pb(
    rs: float,
    temperature: float,
    api_gravity: float,
    gas_sg: float,
) -> float:
    """Bubble point pressure using Petrosky and Farshad correlation (1993).

    Developed for Gulf of Mexico crude oils.

    Args:
        rs: Solution GOR in scf/STB.
        temperature: Temperature in °F.
        api_gravity: Oil API gravity.
        gas_sg: Gas specific gravity.

    Returns:
        Bubble point pressure in psi.
    """
    if rs <= 0:
        return 14.7

    x = (
        4.561e-5 * temperature**1.3911
        - 7.916e-4 * api_gravity**1.5410
    )
    pb = (
        112.727
        * (rs**0.577421 / (gas_sg**0.8439 * 10**x) - 12.340)
    )
    return max(pb, 14.7)


def _petrosky_farshad_bo(
    rs: float,
    temperature: float,
    api_gravity: float,
    gas_sg: float,
) -> float:
    """Oil FVF using Petrosky and Farshad correlation (1993).

    Developed for Gulf of Mexico crude oils.

    Args:
        rs: Solution GOR in scf/STB.
        temperature: Temperature in °F.
        api_gravity: Oil API gravity.
        gas_sg: Gas specific gravity.

    Returns:
        Oil FVF in bbl/STB.
    """
    oil_sg = 141.5 / (api_gravity + 131.5)
    a = (
        rs**0.3738
        * (gas_sg**0.2914 / oil_sg**0.6265)
        + 0.24626 * temperature**0.5371
    ) ** 3.0936
    bo = 1.0113 + 7.2046e-5 * a
    return max(bo, 1.0)


# ---------------------------------------------------------------------------
# Oil density and viscosity
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Gas PVT — Pseudocritical properties
# ---------------------------------------------------------------------------

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


def _piper_mccain_corredor_pseudocritical(
    gas_sg: float,
    h2s_fraction: float = 0.0,
    co2_fraction: float = 0.0,
    n2_fraction: float = 0.0,
) -> tuple[float, float]:
    """Gas pseudocritical properties using Piper, McCain, and Corredor (1993).

    Better than Sutton for gas condensates and gases with significant
    non-hydrocarbon content.

    Args:
        gas_sg: Gas specific gravity (air = 1.0).
        h2s_fraction: Mole fraction of H2S (0-1).
        co2_fraction: Mole fraction of CO2 (0-1).
        n2_fraction: Mole fraction of N2 (0-1).

    Returns:
        Tuple of (Tpc in °R, Ppc in psia).
    """
    # J parameter (Tpc / Ppc)
    j = (
        0.11582
        - 0.45820 * h2s_fraction * (
            -0.90348 + h2s_fraction
        )
        - 0.66026 * co2_fraction * (
            0.03091 + co2_fraction
        )
        - 0.70729 * n2_fraction * (
            -0.42113 + n2_fraction
        )
        - 0.01465 * gas_sg**2
        + 0.20438 * gas_sg
    )

    # K parameter (Tpc / sqrt(Ppc))
    k = (
        3.8216
        - 0.06534 * h2s_fraction * (
            -0.42113 + h2s_fraction
        )
        - 0.42113 * co2_fraction * (
            -0.03691 + co2_fraction
        )
        - 0.91249 * n2_fraction * (
            0.03410 + n2_fraction
        )
        + 17.438 * gas_sg
        - 3.2191 * gas_sg**2
    )

    # Tpc = K^2 / J, Ppc = Tpc / J
    tpc = k**2 / j
    ppc = tpc / j
    return tpc, ppc


# ---------------------------------------------------------------------------
# Gas PVT — Z-factor
# ---------------------------------------------------------------------------

def _hall_yarborough_z(
    temperature: float,
    pressure: float,
    gas_sg: float,
    tpc: float | None = None,
    ppc: float | None = None,
) -> float:
    """Gas Z-factor using the Hall-Yarborough method (1973).

    Uses Newton-Raphson iteration to solve the Hall-Yarborough equation
    of state for the reduced density, then calculates Z.

    Args:
        temperature: Temperature in °F.
        pressure: Pressure in psi.
        gas_sg: Gas specific gravity.
        tpc: Pseudocritical temperature in °R (computed via Sutton if None).
        ppc: Pseudocritical pressure in psia (computed via Sutton if None).

    Returns:
        Gas compressibility factor (Z).
    """
    if tpc is None or ppc is None:
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


def _dranchuk_abou_kassem_z(
    temperature: float,
    pressure: float,
    gas_sg: float,
    pseudocritical_method: str = "sutton",
    h2s_fraction: float = 0.0,
    co2_fraction: float = 0.0,
    n2_fraction: float = 0.0,
) -> float:
    """Gas Z-factor using Dranchuk and Abou-Kassem correlation (1975).

    11-coefficient equation fitted to Standing-Katz chart data.
    Uses Newton-Raphson iteration on reduced density.

    Valid for: 1.0 <= Ppr <= 30, 1.0 <= Tpr <= 3.0

    Args:
        temperature: Temperature in °F.
        pressure: Pressure in psi.
        gas_sg: Gas specific gravity.
        pseudocritical_method: 'sutton' or 'piper' for Tpc/Ppc.
        h2s_fraction: Mole fraction of H2S (for Piper method).
        co2_fraction: Mole fraction of CO2 (for Piper method).
        n2_fraction: Mole fraction of N2 (for Piper method).

    Returns:
        Gas compressibility factor (Z).
    """
    if pseudocritical_method == "piper":
        tpc, ppc = _piper_mccain_corredor_pseudocritical(
            gas_sg, h2s_fraction, co2_fraction, n2_fraction
        )
    else:
        tpc, ppc = _sutton_pseudocritical(gas_sg)

    t_rankine = temperature + 459.67
    t_pr = t_rankine / tpc
    p_pr = pressure / ppc

    # DAK coefficients
    A1, A2, A3, A4, A5 = 0.3265, -1.0700, -0.5339, 0.01569, -0.05165
    A6, A7, A8 = 0.5475, -0.7361, 0.6853
    A9, A10, A11 = 0.1056, 0.6134, 0.7210

    # Newton-Raphson for reduced density rho_r
    # Initial guess: rho_r = 0.27 * Ppr / (Z * Tpr), start with Z=1
    rho_r = 0.27 * p_pr / t_pr  # initial guess (Z=1)

    for _ in range(100):
        rho_r2 = rho_r * rho_r
        rho_r5 = rho_r**5

        # f(rho_r) = 0
        c1 = A1 + A2 / t_pr + A3 / t_pr**3 + A4 / t_pr**4 + A5 / t_pr**5
        c2 = A6 + A7 / t_pr + A8 / t_pr**2
        c3 = A9 * (A7 / t_pr + A8 / t_pr**2)
        c4 = A10 * (1.0 + A11 * rho_r2) * (rho_r2 / t_pr**3) * math.exp(
            -A11 * rho_r2
        )

        f = (
            0.27 * p_pr / (rho_r * t_pr)
            - 1.0
            - c1 * rho_r
            - c2 * rho_r2
            + c3 * rho_r5
            - c4
        )

        # Derivative df/d(rho_r)
        dc4_drho = A10 / t_pr**3 * math.exp(-A11 * rho_r2) * (
            2.0 * rho_r * (1.0 + A11 * rho_r2)
            + rho_r2 * 2.0 * A11 * rho_r
            - (1.0 + A11 * rho_r2) * rho_r2 * 2.0 * A11 * rho_r
        )

        df = (
            -0.27 * p_pr / (rho_r2 * t_pr)
            - c1
            - 2.0 * c2 * rho_r
            + 5.0 * c3 * rho_r**4
            - dc4_drho
        )

        if abs(df) < 1e-30:
            break

        rho_new = rho_r - f / df
        rho_new = max(rho_new, 1e-10)
        rho_new = min(rho_new, 5.0)

        if abs(rho_new - rho_r) < 1e-12:
            rho_r = rho_new
            break
        rho_r = rho_new

    z = 0.27 * p_pr / (rho_r * t_pr) if rho_r > 1e-15 else 1.0
    z = max(z, 0.05)
    return z


# ---------------------------------------------------------------------------
# Gas PVT — FVF, viscosity, compressibility
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Water/Brine PVT — McCain (1990), Osif (1988)
# ---------------------------------------------------------------------------

def _mccain_brine_density(
    temperature: float,
    pressure: float,
    salinity: float,
) -> float:
    """Brine density using McCain correlations (1990).

    First calculates pure water density, then corrects for salinity.

    Args:
        temperature: Temperature in °F.
        pressure: Pressure in psi.
        salinity: Total dissolved solids in ppm (mg/L).

    Returns:
        Brine density in lb/ft³.
    """
    # Pure water density at P and T (lb/ft³)
    # McCain Eq. 2 — pure water density at surface (14.7 psi)
    t = temperature
    rho_w_surface = (
        62.368
        + 0.438603e-1 * t
        + -1.60074e-4 * t**2
    )

    # Pressure correction (simplified from McCain)
    # Water is slightly compressible
    cw_approx = 3.0e-6  # typical water compressibility 1/psi
    rho_w = rho_w_surface * (1.0 + cw_approx * (pressure - 14.7))

    # Salinity correction
    # Convert ppm to weight fraction
    s = salinity / 1e6  # weight fraction
    # McCain salinity correction
    rho_brine = rho_w * (1.0 + 0.695 * s)

    return rho_brine


def _mccain_brine_viscosity(
    temperature: float,
    pressure: float,
    salinity: float,
) -> float:
    """Brine viscosity using McCain correlations (1990).

    First calculates pure water viscosity at T, then applies
    salinity and pressure corrections.

    Args:
        temperature: Temperature in °F.
        pressure: Pressure in psi.
        salinity: Total dissolved solids in ppm.

    Returns:
        Brine viscosity in cp.
    """
    # Pure water viscosity at temperature (cp)
    # McCain (1990), Table 2 — empirical fit for water viscosity at 14.7 psi
    # mu_w = A * T^B  (T in °F)
    # A and B from regression of water viscosity data
    t = temperature
    # Van Wingen (1950) / McCain tabulation:
    # mu_w = exp(1.003 - 1.479e-2*T + 1.982e-5*T^2) for T in °F
    mu_w = math.exp(1.003 - 1.479e-2 * t + 1.982e-5 * t**2)

    # Salinity correction factor
    s = salinity / 1e6  # weight fraction
    # Polynomial correction from McCain
    mu_brine = mu_w * (1.0 + 2.74 * s + 19.05 * s**2)

    # Pressure correction (small effect)
    # mu at pressure = mu at 14.7 * (1 + factor)
    p_corr = 1.0 + 1.0e-6 * (pressure - 14.7) * (
        -0.052 + 0.000267 * temperature
    )
    mu_brine *= max(p_corr, 0.5)

    return max(mu_brine, 0.001)


def _mccain_brine_fvf(
    temperature: float,
    pressure: float,
    salinity: float,
) -> float:
    """Brine formation volume factor using McCain correlation (1990).

    Bw = (1 + dVwT)(1 + dVwP)

    Args:
        temperature: Temperature in °F.
        pressure: Pressure in psi.
        salinity: Total dissolved solids in ppm.

    Returns:
        Brine FVF in bbl/STB.
    """
    t = temperature
    p = pressure

    # Volume change due to temperature (relative to 60°F)
    dt = t - 60.0
    dv_t = (
        -1.0001e-2
        + 1.33391e-4 * dt
        + 5.50654e-7 * dt**2
    )

    # Volume change due to pressure (relative to 14.7 psi)
    dp = p - 14.7
    dv_p = (
        -1.95301e-9 * dp * t
        - 1.72834e-13 * dp**2 * t
        - 3.58922e-7 * dp
        - 2.25341e-10 * dp**2
    )

    bw_fresh = (1.0 + dv_t) * (1.0 + dv_p)

    # Salinity correction
    s = salinity / 1e6  # weight fraction
    # Salinity decreases Bw slightly
    bw = bw_fresh * (1.0 - 0.0753 * s)

    return max(bw, 0.9)


def _osif_brine_compressibility(
    temperature: float,
    pressure: float,
    salinity: float,
) -> float:
    """Brine compressibility using Osif correlation (1988).

    cw = 1 / (7.033*P + 0.5415*S - 537*T + 403300)

    Valid for:
        1000 <= P <= 20000 psi
        200 <= S_nacl_equiv <= 200000 ppm
        200 <= T <= 270 °F

    Args:
        temperature: Temperature in °F.
        pressure: Pressure in psi.
        salinity: NaCl-equivalent salinity in ppm.

    Returns:
        Brine compressibility in 1/psi.
    """
    # Convert ppm to mg/L (assuming density ~ 1 for dilute brines)
    s = salinity / 1e3  # convert to g/L for Osif equation
    denominator = 7.033 * pressure + 0.5415 * s - 537.0 * temperature + 403300.0
    if denominator <= 0:
        # Outside correlation range, return typical value
        return 3.0e-6
    cw = 1.0 / denominator
    return cw


# ===========================================================================
# Public API — all return JSON strings
# ===========================================================================

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
    correlation: str = "standing",
) -> str:
    """Calculate comprehensive black-oil PVT properties at given conditions.

    Computes oil, gas, and fluid properties using published correlations.

    Supported oil correlation sets:
        - 'standing' (default): Standing (1947) for Pb, Rs, Bo
        - 'vasquez_beggs': Vasquez and Beggs (1980) for Rs, Bo
        - 'petrosky_farshad': Petrosky and Farshad (1993) for Pb, Rs, Bo

    Gas and viscosity correlations are always:
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
        correlation: Oil correlation set to use (default 'standing').

    Returns:
        JSON string with all calculated PVT properties, units, and
        correlations used.
    """
    _validate_pvt_inputs(api_gravity, gas_sg, temperature, pressure)

    if separator_pressure <= 0:
        raise ValueError("Separator pressure must be positive (psi)")
    if separator_temperature <= 0:
        raise ValueError("Separator temperature must be positive (°F)")

    valid_correlations = ("standing", "vasquez_beggs", "petrosky_farshad")
    if correlation not in valid_correlations:
        raise ValueError(
            f"Unknown correlation '{correlation}'. "
            f"Choose from: {', '.join(valid_correlations)}"
        )

    # --- Oil properties based on selected correlation ---
    if correlation == "vasquez_beggs":
        rs = _vasquez_beggs_rs(
            pressure, temperature, api_gravity, gas_sg, separator_pressure
        )
        pb = _standing_pb(rs, temperature, api_gravity, gas_sg)
        bo = _vasquez_beggs_bo(
            rs, temperature, api_gravity, gas_sg, separator_pressure
        )
        oil_corr_label = "Vasquez and Beggs (1980)"
        pb_corr_label = "Standing (1947)"
    elif correlation == "petrosky_farshad":
        rs = _petrosky_farshad_rs(pressure, temperature, api_gravity, gas_sg)
        pb = _petrosky_farshad_pb(rs, temperature, api_gravity, gas_sg)
        bo = _petrosky_farshad_bo(rs, temperature, api_gravity, gas_sg)
        oil_corr_label = "Petrosky and Farshad (1993)"
        pb_corr_label = oil_corr_label
    else:
        # Standing (default)
        rs = _standing_rs(pressure, temperature, api_gravity, gas_sg)
        pb = _standing_pb(rs, temperature, api_gravity, gas_sg)
        bo = _standing_bo(rs, temperature, api_gravity, gas_sg)
        oil_corr_label = "Standing (1947)"
        pb_corr_label = oil_corr_label

    # If pressure > Pb, oil is undersaturated: Rs is fixed at Rs(Pb)
    if pressure >= pb:
        rs_at_pb = rs
    else:
        if correlation == "vasquez_beggs":
            rs_at_pb = _vasquez_beggs_rs(
                pb, temperature, api_gravity, gas_sg, separator_pressure
            )
        elif correlation == "petrosky_farshad":
            rs_at_pb = _petrosky_farshad_rs(
                pb, temperature, api_gravity, gas_sg
            )
        else:
            rs_at_pb = _standing_rs(pb, temperature, api_gravity, gas_sg)

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
            "correlation": correlation,
        },
        "oil_properties": {
            "bubble_point_pressure_psi": round(pb, 1),
            "bubble_point_correlation": pb_corr_label,
            "solution_gor_scf_stb": round(rs, 1),
            "solution_gor_correlation": oil_corr_label,
            "oil_fvf_bbl_stb": round(bo, 4),
            "oil_fvf_correlation": oil_corr_label,
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


def calculate_oil_compressibility(
    api_gravity: float,
    gas_sg: float,
    temperature: float,
    pressure: float,
    bubble_point_pressure: float | None = None,
    rs_at_pb: float | None = None,
) -> str:
    """Calculate oil compressibility above and below bubble point.

    Uses Vasquez-Beggs correlation above Pb, and material-balance
    (numerical derivative) approach below Pb.

    Args:
        api_gravity: Oil API gravity (degrees).
        gas_sg: Gas specific gravity (air = 1.0).
        temperature: Reservoir temperature in °F.
        pressure: Current reservoir pressure in psi.
        bubble_point_pressure: Known bubble point pressure in psi (optional;
            estimated via Standing if not provided).
        rs_at_pb: Solution GOR at bubble point in scf/STB (optional;
            estimated via Standing if not provided).

    Returns:
        JSON string with oil compressibility, regime, and inputs.
    """
    _validate_pvt_inputs(api_gravity, gas_sg, temperature, pressure)

    # Estimate Pb and Rs(Pb) if not provided
    if bubble_point_pressure is None or rs_at_pb is None:
        rs_calc = _standing_rs(pressure, temperature, api_gravity, gas_sg)
        pb_calc = _standing_pb(rs_calc, temperature, api_gravity, gas_sg)
        if bubble_point_pressure is None:
            bubble_point_pressure = pb_calc
        if rs_at_pb is None:
            rs_at_pb = _standing_rs(
                bubble_point_pressure, temperature, api_gravity, gas_sg
            )

    if pressure >= bubble_point_pressure:
        # Undersaturated — use Vasquez-Beggs
        co = _vasquez_beggs_co_above_pb(
            temperature, api_gravity, gas_sg, pressure, rs_at_pb
        )
        regime = "undersaturated"
        corr_label = "Vasquez and Beggs (1980)"
    else:
        # Saturated — numerical derivative approach
        rs_p = _standing_rs(pressure, temperature, api_gravity, gas_sg)
        bo_p = _standing_bo(rs_p, temperature, api_gravity, gas_sg)
        z_p = _hall_yarborough_z(temperature, pressure, gas_sg)
        bg_p = _gas_fvf(z_p, temperature, pressure)
        co = _oil_co_below_pb(
            temperature, api_gravity, gas_sg, pressure,
            bubble_point_pressure, bo_p, bg_p
        )
        regime = "saturated"
        corr_label = "Material balance with Standing (1947)"

    result = {
        "oil_compressibility_1_psi": round(co, 8),
        "regime": regime,
        "correlation": corr_label,
        "inputs": {
            "api_gravity": api_gravity,
            "gas_sg": gas_sg,
            "temperature_F": temperature,
            "pressure_psi": pressure,
            "bubble_point_pressure_psi": round(bubble_point_pressure, 1),
            "rs_at_pb_scf_stb": round(rs_at_pb, 1),
        },
        "units": {"oil_compressibility": "1/psi"},
    }
    return json.dumps(result, indent=2)


def calculate_brine_properties(
    temperature: float,
    pressure: float,
    salinity: float = 0.0,
) -> str:
    """Calculate brine (formation water) PVT properties.

    Uses McCain (1990) correlations for density, viscosity, and FVF,
    and Osif (1988) for compressibility.

    Args:
        temperature: Formation temperature in °F.
        pressure: Formation pressure in psi.
        salinity: Total dissolved solids in ppm (default 0 = fresh water).

    Returns:
        JSON string with brine density, viscosity, FVF, compressibility.
    """
    _validate_brine_inputs(temperature, pressure, salinity)

    rho_b = _mccain_brine_density(temperature, pressure, salinity)
    mu_b = _mccain_brine_viscosity(temperature, pressure, salinity)
    bw = _mccain_brine_fvf(temperature, pressure, salinity)
    cw = _osif_brine_compressibility(temperature, pressure, salinity)

    result = {
        "brine_density_lb_ft3": round(rho_b, 2),
        "brine_viscosity_cp": round(mu_b, 4),
        "brine_fvf_bbl_stb": round(bw, 6),
        "brine_compressibility_1_psi": round(cw, 8),
        "correlations": {
            "density": "McCain (1990)",
            "viscosity": "McCain (1990)",
            "fvf": "McCain (1990)",
            "compressibility": "Osif (1988)",
        },
        "inputs": {
            "temperature_F": temperature,
            "pressure_psi": pressure,
            "salinity_ppm": salinity,
        },
        "units": {
            "density": "lb/ft³",
            "viscosity": "cp",
            "fvf": "bbl/STB",
            "compressibility": "1/psi",
        },
    }
    return json.dumps(result, indent=2)


def calculate_gas_z_factor(
    temperature: float,
    pressure: float,
    gas_sg: float,
    method: str = "hall_yarborough",
    pseudocritical_method: str = "sutton",
    h2s_fraction: float = 0.0,
    co2_fraction: float = 0.0,
    n2_fraction: float = 0.0,
) -> str:
    """Calculate gas Z-factor with choice of correlation.

    Args:
        temperature: Temperature in °F.
        pressure: Pressure in psi.
        gas_sg: Gas specific gravity (air = 1.0).
        method: Z-factor method — 'hall_yarborough' or 'dranchuk_abou_kassem'.
        pseudocritical_method: Pseudocritical method — 'sutton' or 'piper'.
        h2s_fraction: Mole fraction of H2S (for Piper method, 0-1).
        co2_fraction: Mole fraction of CO2 (for Piper method, 0-1).
        n2_fraction: Mole fraction of N2 (for Piper method, 0-1).

    Returns:
        JSON string with Z-factor and pseudocritical properties.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive (°F)")
    if pressure <= 0:
        raise ValueError("Pressure must be positive (psi)")
    if gas_sg <= 0:
        raise ValueError("Gas specific gravity must be positive")

    valid_methods = ("hall_yarborough", "dranchuk_abou_kassem")
    if method not in valid_methods:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {', '.join(valid_methods)}"
        )
    valid_pc = ("sutton", "piper")
    if pseudocritical_method not in valid_pc:
        raise ValueError(
            f"Unknown pseudocritical method '{pseudocritical_method}'. "
            f"Choose from: {', '.join(valid_pc)}"
        )

    # Pseudocritical properties
    if pseudocritical_method == "piper":
        tpc, ppc = _piper_mccain_corredor_pseudocritical(
            gas_sg, h2s_fraction, co2_fraction, n2_fraction
        )
        pc_label = "Piper, McCain, and Corredor (1993)"
    else:
        tpc, ppc = _sutton_pseudocritical(gas_sg)
        pc_label = "Sutton (1985)"

    # Z-factor
    if method == "dranchuk_abou_kassem":
        z = _dranchuk_abou_kassem_z(
            temperature, pressure, gas_sg, pseudocritical_method,
            h2s_fraction, co2_fraction, n2_fraction
        )
        z_label = "Dranchuk and Abou-Kassem (1975)"
    else:
        z = _hall_yarborough_z(temperature, pressure, gas_sg, tpc=tpc, ppc=ppc)
        z_label = "Hall and Yarborough (1973)"

    # Gas compressibility
    cg = 1.0 / pressure  # simple ideal-gas approximation for standalone call
    # For better accuracy, compute numerically
    dp = pressure * 0.001
    if method == "dranchuk_abou_kassem":
        z_plus = _dranchuk_abou_kassem_z(
            temperature, pressure + dp, gas_sg, pseudocritical_method,
            h2s_fraction, co2_fraction, n2_fraction
        )
        z_minus = _dranchuk_abou_kassem_z(
            temperature, max(pressure - dp, 1.0), gas_sg,
            pseudocritical_method, h2s_fraction, co2_fraction, n2_fraction
        )
    else:
        z_plus = _hall_yarborough_z(temperature, pressure + dp, gas_sg, tpc=tpc, ppc=ppc)
        z_minus = _hall_yarborough_z(temperature, max(pressure - dp, 1.0), gas_sg, tpc=tpc, ppc=ppc)
    dz_dp = (z_plus - z_minus) / (2.0 * dp)
    cg = 1.0 / pressure - (1.0 / z) * dz_dp

    bg = _gas_fvf(z, temperature, pressure)

    result = {
        "z_factor": round(z, 6),
        "z_factor_correlation": z_label,
        "gas_fvf_rcf_scf": round(bg, 6),
        "gas_compressibility_1_psi": round(cg, 8),
        "pseudocritical_temperature_R": round(tpc, 1),
        "pseudocritical_pressure_psia": round(ppc, 1),
        "pseudocritical_correlation": pc_label,
        "inputs": {
            "temperature_F": temperature,
            "pressure_psi": pressure,
            "gas_sg": gas_sg,
            "method": method,
            "pseudocritical_method": pseudocritical_method,
            "h2s_fraction": h2s_fraction,
            "co2_fraction": co2_fraction,
            "n2_fraction": n2_fraction,
        },
        "units": {
            "z_factor": "dimensionless",
            "gas_fvf": "rcf/scf",
            "gas_compressibility": "1/psi",
            "pseudocritical_temperature": "°R",
            "pseudocritical_pressure": "psia",
        },
    }
    return json.dumps(result, indent=2)
