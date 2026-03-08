"""Comprehensive oilfield unit conversion tool for the petro-mcp server."""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# Conversion registry
# ---------------------------------------------------------------------------
# Each entry maps (from_unit, to_unit) -> conversion callable.
# A callable receives a float and returns a float.
# We also store the *category* so list_units() can group them.

_UNIT_CATEGORIES: dict[str, list[str]] = {
    "volume": [
        "bbl", "m3", "gal", "liters",
        "Mcf", "MMcf", "Bcf",
    ],
    "rate": [
        "bbl/day", "m3/day",
        "Mcf/day", "m3/day",
        "bbl/month",
    ],
    "pressure": ["psi", "kPa", "MPa", "bar", "atm"],
    "length": ["ft", "m", "in", "cm", "miles", "km"],
    "density": ["g/cc", "kg/m3", "lb/ft3", "API", "SG"],
    "temperature": ["F", "C", "K"],
    "permeability": ["md", "m2"],
    "viscosity": ["cp", "Pa.s"],
    "energy": ["BOE", "MMBtu", "Mcf_gas"],
}

# Canonical lowercase lookup for flexible matching
_ALIAS: dict[str, str] = {}


def _register_alias(canonical: str) -> None:
    _ALIAS[canonical.lower()] = canonical


for _units in _UNIT_CATEGORIES.values():
    for _u in _units:
        _register_alias(_u)

# Extra aliases users might type
_EXTRA_ALIASES: dict[str, str] = {
    "barrel": "bbl",
    "barrels": "bbl",
    "gallon": "gal",
    "gallons": "gal",
    "liter": "liters",
    "l": "liters",
    "cubic_meter": "m3",
    "cubic_meters": "m3",
    "feet": "ft",
    "foot": "ft",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "inch": "in",
    "inches": "in",
    "centimeter": "cm",
    "centimeters": "cm",
    "mile": "miles",
    "kilometer": "km",
    "kilometers": "km",
    "kilometre": "km",
    "kilometres": "km",
    "fahrenheit": "F",
    "celsius": "C",
    "centigrade": "C",
    "kelvin": "K",
    "millidarcy": "md",
    "centipoise": "cp",
    "pa.s": "Pa.s",
    "pa·s": "Pa.s",
    "pas": "Pa.s",
    "boe": "BOE",
    "mmbtu": "MMBtu",
    "mcf_gas": "Mcf_gas",
    "api": "API",
    "sg": "SG",
    "specific_gravity": "SG",
    "api_gravity": "API",
    "bbl/d": "bbl/day",
    "bopd": "bbl/day",
    "m3/d": "m3/day",
    "mcf/d": "Mcf/day",
    "mcfd": "Mcf/day",
    "bbl/mon": "bbl/month",
    "bbl/mo": "bbl/month",
}
for _alias, _canon in _EXTRA_ALIASES.items():
    _ALIAS[_alias.lower()] = _canon

# ---------------------------------------------------------------------------
# Conversion factors & functions
# ---------------------------------------------------------------------------

# Volume constants
_BBL_TO_M3 = 0.158987294928
_BBL_TO_GAL = 42.0
_BBL_TO_LITERS = 158.987294928
_MCF_TO_M3 = 28.316846592  # 1 Mcf = 1000 ft3, 1 ft3 = 0.028316846592 m3

# Pressure constants
_PSI_TO_KPA = 6.894757293168
_PSI_TO_MPA = 0.006894757293168
_PSI_TO_BAR = 0.0689475729316836
_PSI_TO_ATM = 0.068045963169952

# Length constants
_FT_TO_M = 0.3048
_IN_TO_CM = 2.54
_MILES_TO_KM = 1.609344

# Density
_GCC_TO_KGM3 = 1000.0
_GCC_TO_LBFT3 = 62.427960576145

# Permeability
_MD_TO_M2 = 9.869233e-16

# Viscosity
_CP_TO_PAS = 0.001

# Energy / BOE
_BOE_TO_MMBTU = 5.8
_BOE_TO_MCF_GAS = 6.0

# Rate
_DAYS_PER_MONTH = 30.44


def _build_linear(factor: float):
    """Return a pair of lambdas for forward/reverse linear conversion."""
    return (lambda v: v * factor), (lambda v: v / factor)


# Map of (from, to) -> callable
_CONVERSIONS: dict[tuple[str, str], callable] = {}


def _register_pair(u1: str, u2: str, forward, reverse=None):
    """Register a bidirectional conversion."""
    _CONVERSIONS[(u1, u2)] = forward
    if reverse is not None:
        _CONVERSIONS[(u2, u1)] = reverse


def _reg_linear(u1: str, u2: str, factor: float):
    fwd, rev = _build_linear(factor)
    _register_pair(u1, u2, fwd, rev)


# --- Volume ---
_reg_linear("bbl", "m3", _BBL_TO_M3)
_reg_linear("bbl", "gal", _BBL_TO_GAL)
_reg_linear("bbl", "liters", _BBL_TO_LITERS)
_reg_linear("Mcf", "m3", _MCF_TO_M3)
_reg_linear("MMcf", "m3", _MCF_TO_M3 * 1000)
_reg_linear("Bcf", "m3", _MCF_TO_M3 * 1_000_000)
# cross gas-volume conversions
_reg_linear("Mcf", "MMcf", 0.001)
_reg_linear("Mcf", "Bcf", 1e-6)
_reg_linear("MMcf", "Bcf", 0.001)
_reg_linear("gal", "liters", _BBL_TO_LITERS / _BBL_TO_GAL)
_reg_linear("m3", "liters", 1000.0)
_reg_linear("m3", "gal", 1.0 / _BBL_TO_M3 * _BBL_TO_GAL)

# --- Rate ---
_reg_linear("bbl/day", "m3/day", _BBL_TO_M3)
_reg_linear("Mcf/day", "m3/day", _MCF_TO_M3)
_reg_linear("bbl/day", "bbl/month", _DAYS_PER_MONTH)

# --- Pressure ---
_reg_linear("psi", "kPa", _PSI_TO_KPA)
_reg_linear("psi", "MPa", _PSI_TO_MPA)
_reg_linear("psi", "bar", _PSI_TO_BAR)
_reg_linear("psi", "atm", _PSI_TO_ATM)
# Cross-pressure shortcuts
_reg_linear("kPa", "MPa", 0.001)
_reg_linear("bar", "atm", _PSI_TO_ATM / _PSI_TO_BAR)
_reg_linear("bar", "kPa", _PSI_TO_KPA / _PSI_TO_BAR)
_reg_linear("bar", "MPa", _PSI_TO_MPA / _PSI_TO_BAR)
_reg_linear("kPa", "atm", _PSI_TO_ATM / _PSI_TO_KPA)
_reg_linear("MPa", "atm", _PSI_TO_ATM / _PSI_TO_MPA)

# --- Length ---
_reg_linear("ft", "m", _FT_TO_M)
_reg_linear("in", "cm", _IN_TO_CM)
_reg_linear("miles", "km", _MILES_TO_KM)
_reg_linear("ft", "in", 12.0)
_reg_linear("m", "cm", 100.0)
_reg_linear("ft", "cm", _FT_TO_M * 100)
_reg_linear("in", "m", _IN_TO_CM / 100)
_reg_linear("ft", "km", _FT_TO_M / 1000)
_reg_linear("miles", "m", _MILES_TO_KM * 1000)
_reg_linear("m", "km", 0.001)

# --- Density (linear portion) ---
_reg_linear("g/cc", "kg/m3", _GCC_TO_KGM3)
_reg_linear("g/cc", "lb/ft3", _GCC_TO_LBFT3)
_reg_linear("kg/m3", "lb/ft3", _GCC_TO_LBFT3 / _GCC_TO_KGM3)

# API <-> SG (non-linear)
_register_pair(
    "API", "SG",
    lambda api: 141.5 / (api + 131.5),
    lambda sg: 141.5 / sg - 131.5,
)
# API <-> g/cc  (SG is dimensionless relative to water at 60F => same as g/cc)
_register_pair(
    "API", "g/cc",
    lambda api: 141.5 / (api + 131.5),
    lambda gcc: 141.5 / gcc - 131.5,
)
_register_pair(
    "API", "kg/m3",
    lambda api: 141.5 / (api + 131.5) * _GCC_TO_KGM3,
    lambda kgm3: 141.5 / (kgm3 / _GCC_TO_KGM3) - 131.5,
)
_register_pair(
    "API", "lb/ft3",
    lambda api: 141.5 / (api + 131.5) * _GCC_TO_LBFT3,
    lambda lbft3: 141.5 / (lbft3 / _GCC_TO_LBFT3) - 131.5,
)
_register_pair(
    "SG", "g/cc",
    lambda sg: sg,          # SG is relative density, numerically == g/cc for water ref
    lambda gcc: gcc,
)
_register_pair(
    "SG", "kg/m3",
    lambda sg: sg * _GCC_TO_KGM3,
    lambda kgm3: kgm3 / _GCC_TO_KGM3,
)
_register_pair(
    "SG", "lb/ft3",
    lambda sg: sg * _GCC_TO_LBFT3,
    lambda lbft3: lbft3 / _GCC_TO_LBFT3,
)

# --- Temperature (non-linear) ---
_register_pair("F", "C", lambda f: (f - 32) * 5 / 9, lambda c: c * 9 / 5 + 32)
_register_pair("F", "K", lambda f: (f - 32) * 5 / 9 + 273.15, lambda k: (k - 273.15) * 9 / 5 + 32)
_register_pair("C", "K", lambda c: c + 273.15, lambda k: k - 273.15)

# --- Permeability ---
_reg_linear("md", "m2", _MD_TO_M2)

# --- Viscosity ---
_reg_linear("cp", "Pa.s", _CP_TO_PAS)

# --- Energy / BOE ---
_reg_linear("BOE", "MMBtu", _BOE_TO_MMBTU)
_reg_linear("BOE", "Mcf_gas", _BOE_TO_MCF_GAS)
_reg_linear("MMBtu", "Mcf_gas", _BOE_TO_MCF_GAS / _BOE_TO_MMBTU)

# Identity conversions
for _units in _UNIT_CATEGORIES.values():
    for _u in _units:
        _CONVERSIONS[(_u, _u)] = lambda v: v


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _normalize_unit(unit: str) -> str:
    """Resolve aliases to canonical unit names."""
    key = unit.strip().lower()
    if key in _ALIAS:
        return _ALIAS[key]
    # Try original (case-sensitive match)
    for cat_units in _UNIT_CATEGORIES.values():
        if unit in cat_units:
            return unit
    raise ValueError(
        f"Unknown unit: '{unit}'. Use list_units() to see supported units."
    )


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between oilfield / petroleum engineering units.

    Supports volume, rate, pressure, length, density (including API gravity),
    temperature, permeability, viscosity, and energy/BOE conversions.

    Args:
        value: Numeric value to convert.
        from_unit: Source unit (e.g. 'bbl', 'psi', 'API').
        to_unit: Target unit (e.g. 'm3', 'kPa', 'SG').

    Returns:
        JSON string with value, from_unit, to_unit, result, and
        conversion_factor (where applicable).
    """
    src = _normalize_unit(from_unit)
    dst = _normalize_unit(to_unit)

    key = (src, dst)
    if key not in _CONVERSIONS:
        raise ValueError(
            f"No conversion registered from '{src}' to '{dst}'. "
            f"Use list_units() to see supported units and categories."
        )

    converter = _CONVERSIONS[key]
    result_value = converter(value)

    # Compute a conversion factor for linear conversions (where factor = result/value).
    # For non-linear conversions (temperature, API/SG) we report None.
    if value != 0:
        factor = result_value / value
    else:
        # Try with a probe value to get factor
        try:
            probe = converter(1.0)
            factor = probe
        except ZeroDivisionError:
            factor = None

    # Detect non-linear: check if factor is consistent at another point
    is_linear = True
    try:
        if value != 0:
            probe2 = converter(value * 2)
            if abs(probe2 - result_value * 2) > abs(result_value) * 1e-9 + 1e-12:
                is_linear = False
        else:
            probe2 = converter(2.0)
            if factor is not None and abs(probe2 - factor * 2) > abs(factor) * 1e-9 + 1e-12:
                is_linear = False
    except (ZeroDivisionError, OverflowError):
        is_linear = False

    def _smart_round(v: float, digits: int = 10) -> float:
        """Round to *digits* significant figures for very small/large numbers."""
        if v == 0:
            return 0.0
        import math as _m
        magnitude = _m.floor(_m.log10(abs(v)))
        if magnitude < -digits or magnitude > digits:
            # Use significant-figure rounding
            return round(v, -int(magnitude) + digits - 1)
        return round(v, digits)

    output = {
        "value": value,
        "from_unit": src,
        "to_unit": dst,
        "result": _smart_round(result_value),
        "conversion_factor": _smart_round(factor) if (is_linear and factor is not None) else None,
    }

    return json.dumps(output, indent=2)


def list_units() -> str:
    """Return all supported unit categories and their units.

    Returns:
        JSON string mapping each category name to its list of supported units.
    """
    return json.dumps(_UNIT_CATEGORIES, indent=2)
