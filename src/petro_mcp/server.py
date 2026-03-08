"""Main MCP server entry point for petro-mcp."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from petro_mcp.prompts.templates import TEMPLATES
from petro_mcp.resources.well_data import list_wells, production_summary
from petro_mcp.tools.calculations import calculate_ip_ratio, nodal_analysis
from petro_mcp.tools.decline import calculate_eur, fit_decline_curve
from petro_mcp.tools.las import get_curve_data, get_well_header, list_curves, read_las_file
from petro_mcp.tools.production import query_production_data
from petro_mcp.tools.pvt import bubble_point as _bubble_point, calculate_pvt as _calculate_pvt
from petro_mcp.tools.trends import analyze_production_trends
from petro_mcp.tools.units import convert_units as _convert_units, list_units as _list_units

mcp = FastMCP(
    "petro-mcp",
    description="Petroleum engineering data and tools for LLMs",
)


# --- LAS File Tools ---


@mcp.tool()
def read_las(file_path: str) -> str:
    """Parse a LAS 2.0 well log file and return header info and curve data summary.

    Args:
        file_path: Absolute path to the LAS file.
    """
    return read_las_file(file_path)


@mcp.tool()
def get_header(file_path: str) -> str:
    """Extract well header metadata (well name, UWI, location, KB, TD, etc.) from a LAS file.

    Args:
        file_path: Absolute path to the LAS file.
    """
    return get_well_header(file_path)


@mcp.tool()
def get_curves(file_path: str) -> str:
    """List all curves in a LAS file with their units and descriptions.

    Args:
        file_path: Absolute path to the LAS file.
    """
    return list_curves(file_path)


@mcp.tool()
def get_curve_values(
    file_path: str,
    curve_names: list[str],
    start_depth: float | None = None,
    end_depth: float | None = None,
) -> str:
    """Get specific curve data from a LAS file with optional depth range filtering.

    Args:
        file_path: Absolute path to the LAS file.
        curve_names: List of curve mnemonics to retrieve (e.g., ["GR", "RHOB"]).
        start_depth: Optional start depth for filtering.
        end_depth: Optional end depth for filtering.
    """
    return get_curve_data(file_path, curve_names, start_depth, end_depth)


# --- Production Data Tools ---


@mcp.tool()
def query_production(
    file_path: str,
    well_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Query production data from a CSV file (columns: date, well_name, oil, gas, water).

    Args:
        file_path: Absolute path to the production CSV file.
        well_name: Optional well name to filter by.
        start_date: Optional start date (YYYY-MM-DD).
        end_date: Optional end date (YYYY-MM-DD).
    """
    return query_production_data(file_path, well_name, start_date, end_date)


@mcp.tool(name="analyze_trends")
def analyze_trends(
    file_path: str,
    well_name: str | None = None,
) -> str:
    """Analyze production trends and detect anomalies (shut-ins, rate jumps, water breakthrough, GOR blowouts).

    Computes per-well water cut trend, GOR trend, oil decline rate,
    cumulative production, and flags anomalous events.

    Args:
        file_path: Absolute path to the production CSV file.
        well_name: Optional well name to filter by.
    """
    return analyze_production_trends(file_path, well_name)


# --- Decline Curve Tools ---


@mcp.tool()
def fit_decline(
    production_data: list[dict[str, float]],
    model: str = "hyperbolic",
) -> str:
    """Fit decline curves to production data.

    Supports Arps models (exponential, hyperbolic, harmonic), modified
    hyperbolic with Dmin terminal decline switch, and Duong model for
    unconventional/shale wells with fracture-dominated flow.

    Returns fitted parameters, R-squared, and predicted rates.
    Physics-constrained: b-factor bounded to [0, 2], non-negative rates enforced.

    Args:
        production_data: List of dicts with 'time' (months) and 'rate' keys,
            or 'oil'/'gas' keys (time assumed as sequential months).
        model: Decline model - 'exponential', 'hyperbolic', 'harmonic',
            'modified_hyperbolic', or 'duong'.
    """
    return fit_decline_curve(production_data, model)


@mcp.tool(name="calculate_eur")
def calculate_eur_tool(
    qi: float,
    Di: float = 0.0,
    b: float = 0.0,
    economic_limit: float = 5.0,
    model: str = "hyperbolic",
    Dmin: float = 0.005,
    a: float = 1.0,
    m: float = 1.1,
) -> str:
    """Calculate Estimated Ultimate Recovery using decline parameters.

    Supports Arps models (exponential, hyperbolic, harmonic), modified
    hyperbolic with Dmin terminal decline switch, and Duong model for
    unconventional/shale wells.

    Args:
        qi: Initial production rate (bbl/day or Mcf/day).
        Di: Initial decline rate (1/month, nominal). Used by exponential,
            hyperbolic, harmonic, and modified_hyperbolic models.
        b: Arps b-factor (0=exponential, 1=harmonic, 0-2=hyperbolic).
        economic_limit: Minimum economic rate (same units as qi).
        model: Decline model - 'exponential', 'hyperbolic', 'harmonic',
            'modified_hyperbolic', or 'duong'.
        Dmin: Minimum terminal decline rate for modified_hyperbolic (1/month).
        a: Duong intercept parameter (typically 0.5-2.0).
        m: Duong slope parameter (typically 1.0-1.5).
    """
    return calculate_eur(qi, Di, b, economic_limit, model, Dmin=Dmin, a=a, m=m)


# --- Calculation Tools ---


@mcp.tool()
def calculate_ratios(
    oil_rate: float,
    gas_rate: float,
    water_rate: float,
) -> str:
    """Calculate producing ratios: GOR, WOR, water cut, and classify well type.

    Args:
        oil_rate: Oil rate in bbl/day (BOPD).
        gas_rate: Gas rate in Mcf/day.
        water_rate: Water rate in bbl/day (BWPD).
    """
    return calculate_ip_ratio(oil_rate, gas_rate, water_rate)


@mcp.tool()
def run_nodal_analysis(
    reservoir_pressure: float,
    PI: float,
    tubing_size: float,
    wellhead_pressure: float,
    depth: float = 8000.0,
    fluid_gradient: float = 0.35,
) -> str:
    """Simplified nodal analysis: find IPR/VLP intersection for operating point.

    Uses Vogel IPR and simplified vertical lift model.

    Args:
        reservoir_pressure: Average reservoir pressure in psi.
        PI: Productivity index in bbl/day/psi.
        tubing_size: Tubing inner diameter in inches.
        wellhead_pressure: Wellhead flowing pressure in psi.
        depth: True vertical depth in feet (default 8000).
        fluid_gradient: Fluid pressure gradient in psi/ft (default 0.35).
    """
    return nodal_analysis(reservoir_pressure, PI, tubing_size, wellhead_pressure, depth, fluid_gradient)


# --- PVT Tools ---


@mcp.tool()
def calculate_pvt_properties(
    api_gravity: float,
    gas_sg: float,
    temperature: float,
    pressure: float,
    separator_pressure: float = 100.0,
    separator_temperature: float = 60.0,
) -> str:
    """Calculate comprehensive black-oil PVT properties at given conditions.

    Returns bubble point, solution GOR, oil FVF, oil density, oil viscosity,
    gas Z-factor, gas FVF, gas viscosity, and gas compressibility using
    Standing, Beggs-Robinson, Hall-Yarborough, and Lee-Gonzalez-Eakin
    correlations.

    Args:
        api_gravity: Oil API gravity (degrees).
        gas_sg: Gas specific gravity (air = 1.0).
        temperature: Reservoir temperature in °F.
        pressure: Current reservoir pressure in psi.
        separator_pressure: Separator pressure in psi (default 100).
        separator_temperature: Separator temperature in °F (default 60).
    """
    return _calculate_pvt(
        api_gravity, gas_sg, temperature, pressure,
        separator_pressure, separator_temperature,
    )


@mcp.tool()
def calculate_bubble_point(
    api_gravity: float,
    gas_sg: float,
    temperature: float,
    rs: float,
) -> str:
    """Calculate bubble point pressure using Standing's correlation (1947).

    Args:
        api_gravity: Oil API gravity (degrees).
        gas_sg: Gas specific gravity (air = 1.0).
        temperature: Reservoir temperature in °F.
        rs: Solution gas-oil ratio at bubble point in scf/STB.
    """
    return _bubble_point(api_gravity, gas_sg, temperature, rs)


# --- Unit Conversion Tools ---


@mcp.tool()
def convert_oilfield_units(
    value: float,
    from_unit: str,
    to_unit: str,
) -> str:
    """Convert between oilfield and SI units.

    Supports volume (bbl, m3, gal, liters, Mcf, MMcf, Bcf), rate (bbl/day,
    m3/day, Mcf/day, bbl/month), pressure (psi, kPa, MPa, bar, atm), length
    (ft, m, in, cm, miles, km), density (g/cc, kg/m3, lb/ft3, API gravity,
    SG), temperature (F, C, K), permeability (md, m2), viscosity (cp, Pa.s),
    and energy/BOE (BOE, MMBtu, Mcf_gas).

    Args:
        value: Numeric value to convert.
        from_unit: Source unit (e.g. 'bbl', 'psi', 'API').
        to_unit: Target unit (e.g. 'm3', 'kPa', 'SG').
    """
    return _convert_units(value, from_unit, to_unit)


@mcp.tool()
def list_oilfield_units() -> str:
    """List all supported oilfield unit categories and their units."""
    return _list_units()


# --- Resources ---


@mcp.resource("wells://list/{directory}")
def browse_wells(directory: str) -> str:
    """Browse wells in a directory of LAS and CSV files."""
    return list_wells(directory)


@mcp.resource("wells://production/{file_path}")
def get_production_summary(file_path: str) -> str:
    """Get production summary for all wells in a CSV file."""
    return production_summary(file_path)


# --- Prompts ---


@mcp.prompt()
def analyze_decline() -> str:
    """Analyze a well's production decline behavior and estimate EUR."""
    return TEMPLATES["analyze_decline"]["template"]


@mcp.prompt()
def compare_completions() -> str:
    """Compare completion effectiveness across multiple wells."""
    return TEMPLATES["compare_completions"]["template"]


@mcp.prompt()
def summarize_logs() -> str:
    """Summarize the log curves in a LAS well log file."""
    return TEMPLATES["summarize_logs"]["template"]


@mcp.prompt()
def production_anomalies() -> str:
    """Detect anomalies and changes in production patterns."""
    return TEMPLATES["production_anomalies"]["template"]


@mcp.prompt()
def calculate_well_eur() -> str:
    """Calculate Estimated Ultimate Recovery for a well."""
    return TEMPLATES["calculate_well_eur"]["template"]


def main():
    """Run the petro-mcp server."""
    mcp.run()


if __name__ == "__main__":
    main()
