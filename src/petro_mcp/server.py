"""Main MCP server entry point for petro-mcp."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Tool group registry
# ---------------------------------------------------------------------------

TOOL_GROUPS: dict[str, str] = {
    "las": "LAS well log parsing",
    "production": "Production data loader",
    "decline": "Decline curve analysis (Arps)",
    "pvt": "PVT correlations",
    "petrophysics": "Petrophysics interpretation",
    "reservoir": "Reservoir engineering (volumetrics, MBE, p/z)",
    "units": "Unit conversions",
}


# ---------------------------------------------------------------------------
# Per-group registration functions
# ---------------------------------------------------------------------------


def _register_las(mcp: FastMCP) -> None:
    from petro_mcp.tools.las import get_curve_data, read_las_file

    @mcp.tool()
    def read_las(file_path: str) -> str:
        """Parse a LAS 2.0 well log file and return header, curves list, and data summary.

        Args:
            file_path: Absolute path to the LAS file.
        """
        return read_las_file(file_path)

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



def _register_production(mcp: FastMCP) -> None:
    from petro_mcp.tools.production import query_production_data

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


def _register_decline(mcp: FastMCP) -> None:
    from petro_mcp.tools.decline import calculate_eur, fit_decline_curve

    @mcp.tool()
    def fit_decline(
        production_data: list[dict[str, float]],
        model: str = "hyperbolic",
    ) -> str:
        """Fit an Arps decline curve to production data.

        Returns fitted parameters, R-squared, and predicted rates.
        Physics-constrained: b-factor bounded to [0, 2], non-negative rates enforced.

        Args:
            production_data: List of dicts with 'time' (months) and 'rate' keys,
                or 'oil'/'gas' keys (time assumed as sequential months).
            model: Arps decline model - 'exponential', 'hyperbolic', or 'harmonic'.
        """
        return fit_decline_curve(production_data, model)

    @mcp.tool(name="calculate_eur")
    def calculate_eur_tool(
        qi: float,
        Di: float = 0.0,
        b: float = 0.0,
        economic_limit: float = 5.0,
        model: str = "hyperbolic",
    ) -> str:
        """Calculate Estimated Ultimate Recovery from Arps decline parameters.

        Args:
            qi: Initial production rate (bbl/day or Mcf/day).
            Di: Initial decline rate (1/month, nominal).
            b: Arps b-factor (0=exponential, 1=harmonic, 0-2=hyperbolic).
            economic_limit: Minimum economic rate (same units as qi).
            model: Arps model - 'exponential', 'hyperbolic', or 'harmonic'.
        """
        return calculate_eur(qi, Di, b, economic_limit, model)



def _register_reservoir(mcp: FastMCP) -> None:
    from petro_mcp.tools.reservoir import (
        calculate_pz_analysis as _pz_analysis,
        calculate_recovery_factor as _recovery_factor,
        calculate_volumetric_ogip as _volumetric_ogip,
        calculate_volumetric_ooip as _volumetric_ooip,
    )

    @mcp.tool()
    def pz_analysis(
        pressures: list[float],
        cumulative_gas: list[float],
        abandonment_pressure: float | None = None,
    ) -> str:
        """Gas material balance: P/Z vs cumulative gas production analysis.

        Fits a linear P/Z vs Gp trend to estimate Original Gas In Place (OGIP).
        The #1 reservoir engineering spreadsheet calculation.

        Args:
            pressures: Reservoir pressures (or P/Z values) in psi at each time step.
            cumulative_gas: Cumulative gas production in Bcf at each pressure.
            abandonment_pressure: Optional abandonment pressure in psi for
                recoverable gas estimate.
        """
        return _pz_analysis(pressures, cumulative_gas, abandonment_pressure)

    @mcp.tool()
    def volumetric_ooip(
        area_acres: float,
        thickness_ft: float,
        porosity: float,
        sw: float,
        bo: float,
    ) -> str:
        """Calculate volumetric Original Oil In Place (OOIP).

        OOIP = 7758 * A * h * phi * (1-Sw) / Bo (STB)

        Args:
            area_acres: Reservoir area in acres.
            thickness_ft: Net pay thickness in feet.
            porosity: Porosity (fraction, 0-1).
            sw: Water saturation (fraction, 0-1).
            bo: Oil formation volume factor (bbl/STB).
        """
        return _volumetric_ooip(area_acres, thickness_ft, porosity, sw, bo)

    @mcp.tool()
    def volumetric_ogip(
        area_acres: float,
        thickness_ft: float,
        porosity: float,
        sw: float,
        bg: float,
    ) -> str:
        """Calculate volumetric Original Gas In Place (OGIP).

        OGIP = 43560 * A * h * phi * (1-Sw) / Bg (scf)

        Args:
            area_acres: Reservoir area in acres.
            thickness_ft: Net pay thickness in feet.
            porosity: Porosity (fraction, 0-1).
            sw: Water saturation (fraction, 0-1).
            bg: Gas formation volume factor (ft3/scf).
        """
        return _volumetric_ogip(area_acres, thickness_ft, porosity, sw, bg)

    @mcp.tool()
    def recovery_factor(
        ooip_or_ogip: float,
        cumulative_production: float,
    ) -> str:
        """Calculate recovery factor (RF = Np/N or Gp/G).

        Works for both oil and gas -- just use consistent units.

        Args:
            ooip_or_ogip: Original oil or gas in place.
            cumulative_production: Cumulative production (same units).
        """
        return _recovery_factor(ooip_or_ogip, cumulative_production)

def _register_pvt(mcp: FastMCP) -> None:
    from petro_mcp.tools.pvt import (
        bubble_point as _bubble_point,
        calculate_gas_z_factor as _calculate_gas_z,
        calculate_pvt as _calculate_pvt,
    )

    @mcp.tool()
    def calculate_pvt_properties(
        api_gravity: float,
        gas_sg: float,
        temperature: float,
        pressure: float,
        separator_pressure: float = 100.0,
        correlation: str = "standing",
    ) -> str:
        """Calculate comprehensive black-oil PVT properties at given conditions.

        Returns bubble point, solution GOR, oil FVF, oil density, oil viscosity,
        gas Z-factor, gas FVF, gas viscosity, and gas compressibility.

        Supported oil correlation sets:
            - 'standing' (default): Standing (1947)
            - 'vasquez_beggs': Vasquez and Beggs (1980)
            - 'petrosky_farshad': Petrosky and Farshad (1993)

        Args:
            api_gravity: Oil API gravity (degrees).
            gas_sg: Gas specific gravity (air = 1.0).
            temperature: Reservoir temperature in F.
            pressure: Current reservoir pressure in psi.
            separator_pressure: Separator pressure in psi (default 100).
            correlation: Oil correlation set -- 'standing', 'vasquez_beggs',
                or 'petrosky_farshad'.
        """
        return _calculate_pvt(
            api_gravity, gas_sg, temperature, pressure,
            separator_pressure, correlation,
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
            temperature: Reservoir temperature in F.
            rs: Solution gas-oil ratio at bubble point in scf/STB.
        """
        return _bubble_point(api_gravity, gas_sg, temperature, rs)

    @mcp.tool()
    def calculate_gas_z(
        temperature: float,
        pressure: float,
        gas_sg: float,
        method: str = "hall_yarborough",
        pseudocritical_method: str = "sutton",
        h2s_fraction: float = 0.0,
        co2_fraction: float = 0.0,
        n2_fraction: float = 0.0,
    ) -> str:
        """Calculate gas Z-factor with choice of correlation and pseudocritical method.

        Z-factor methods: 'hall_yarborough' (default), 'dranchuk_abou_kassem'.
        Pseudocritical methods: 'sutton' (default), 'piper' (better for gas
        condensates and sour gases).

        Args:
            temperature: Temperature in F.
            pressure: Pressure in psi.
            gas_sg: Gas specific gravity (air = 1.0).
            method: Z-factor correlation -- 'hall_yarborough' or 'dranchuk_abou_kassem'.
            pseudocritical_method: Pseudocritical method -- 'sutton' or 'piper'.
            h2s_fraction: Mole fraction of H2S (for Piper method, 0-1).
            co2_fraction: Mole fraction of CO2 (for Piper method, 0-1).
            n2_fraction: Mole fraction of N2 (for Piper method, 0-1).
        """
        return _calculate_gas_z(
            temperature, pressure, gas_sg, method, pseudocritical_method,
            h2s_fraction, co2_fraction, n2_fraction,
        )


def _register_petrophysics(mcp: FastMCP) -> None:
    from petro_mcp.tools.petrophysics import (
        calculate_archie_sw as _archie_sw,
        calculate_density_porosity as _density_porosity,
        calculate_net_pay as _net_pay,
        calculate_vshale as _vshale,
    )

    @mcp.tool()
    def calculate_vshale(
        gr: float, gr_clean: float, gr_shale: float, method: str = "linear",
    ) -> str:
        """Calculate shale volume (Vshale) from gamma ray log.

        Methods: linear, larionov_tertiary, larionov_older, clavier.

        Args:
            gr: Gamma ray reading (API units).
            gr_clean: GR in clean sand (API units).
            gr_shale: GR in pure shale (API units).
            method: Calculation method. Default 'linear'.
        """
        return _vshale(gr, gr_clean, gr_shale, method)

    @mcp.tool()
    def calculate_density_porosity(
        rhob: float, rho_matrix: float = 2.65, rho_fluid: float = 1.0,
    ) -> str:
        """Calculate porosity from bulk density log.

        Args:
            rhob: Bulk density (g/cc).
            rho_matrix: Matrix density (g/cc). Default 2.65 (sandstone).
            rho_fluid: Fluid density (g/cc). Default 1.0 (fresh water).
        """
        return _density_porosity(rhob, rho_matrix, rho_fluid)

    @mcp.tool()
    def calculate_archie_sw(
        rt: float, phi: float, rw: float,
        a: float = 1.0, m: float = 2.0, n: float = 2.0,
    ) -> str:
        """Calculate water saturation using Archie equation (clean sands).

        Sw = (a * Rw / (phi^m * Rt))^(1/n)

        Args:
            rt: True formation resistivity (ohm-m).
            phi: Porosity (fraction v/v, 0-1).
            rw: Formation water resistivity (ohm-m).
            a: Tortuosity factor. Default 1.0.
            m: Cementation exponent. Default 2.0.
            n: Saturation exponent. Default 2.0.
        """
        return _archie_sw(rt, phi, rw, a, m, n)

    @mcp.tool()
    def calculate_net_pay(
        depths: list[float], phi: list[float], sw: list[float], vshale: list[float],
        phi_cutoff: float = 0.06, sw_cutoff: float = 0.5, vsh_cutoff: float = 0.5,
    ) -> str:
        """Determine net pay by applying porosity, Sw, and Vshale cutoffs to log data.

        Returns net pay thickness, net-to-gross, average properties over pay,
        and per-sample pay flags.

        Args:
            depths: Measured depths (ft).
            phi: Porosity values (fraction v/v) at each depth.
            sw: Water saturation values (fraction v/v) at each depth.
            vshale: Shale volume values (fraction v/v) at each depth.
            phi_cutoff: Minimum porosity for pay. Default 0.06.
            sw_cutoff: Maximum water saturation for pay. Default 0.5.
            vsh_cutoff: Maximum Vshale for pay. Default 0.5.
        """
        return _net_pay(depths, phi, sw, vshale, phi_cutoff, sw_cutoff, vsh_cutoff)

def _register_units(mcp: FastMCP) -> None:
    from petro_mcp.tools.units import convert_units as _convert_units, list_units as _list_units

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


# ---------------------------------------------------------------------------
# Mapping from group name -> registration function
# ---------------------------------------------------------------------------

_GROUP_REGISTRARS: dict[str, callable] = {
    "las": _register_las,
    "production": _register_production,
    "decline": _register_decline,
    "reservoir": _register_reservoir,
    "pvt": _register_pvt,
    "petrophysics": _register_petrophysics,
    "units": _register_units,
}


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def create_server(groups: set[str] | None = None) -> FastMCP:
    """Create an MCP server with selected tool groups.

    Args:
        groups: Set of tool group names to load.  ``None`` means all groups.

    Returns:
        A configured :class:`FastMCP` instance ready to run.
    """
    mcp = FastMCP(
        "petro-mcp",
        instructions="Petroleum engineering calculations for LLMs",
    )

    if groups is None:
        groups = set(TOOL_GROUPS.keys())

    for name in groups:
        registrar = _GROUP_REGISTRARS.get(name)
        if registrar is not None:
            registrar(mcp)

    return mcp


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """Run the petro-mcp server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="petro-mcp: Petroleum engineering MCP server",
    )
    parser.add_argument(
        "--tools",
        type=str,
        default=None,
        help="Comma-separated tool groups to load (default: all). "
             "Available: " + ", ".join(sorted(TOOL_GROUPS.keys())),
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available tool groups and exit",
    )
    args = parser.parse_args()

    if args.list_tools:
        for name, desc in sorted(TOOL_GROUPS.items()):
            print(f"  {name:20s} {desc}")
        return

    groups = None
    if args.tools:
        groups = {g.strip() for g in args.tools.split(",")}
        invalid = groups - set(TOOL_GROUPS.keys())
        if invalid:
            parser.error(f"Unknown tool groups: {invalid}")

    server = create_server(groups)
    server.run()


if __name__ == "__main__":
    main()
