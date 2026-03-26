"""Main MCP server entry point for petro-mcp."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from petro_mcp.prompts.templates import TEMPLATES
from petro_mcp.resources.well_data import list_wells, production_summary

# ---------------------------------------------------------------------------
# Tool group registry
# ---------------------------------------------------------------------------

TOOL_GROUPS: dict[str, str] = {
    "las": "LAS well log parsing",
    "production": "Production data and trends",
    "decline": "Decline curve analysis (Arps + advanced)",
    "pvt": "PVT correlations",
    "petrophysics": "Petrophysics interpretation",
    "calculations": "Ratios and nodal analysis",
    "units": "Unit conversions",
    "drilling": "Drilling calculations",
    "reservoir": "Reservoir engineering",
    "economics": "Production economics",
    "trajectory": "Well trajectory (requires welleng)",
    "production_eng": "Production engineering",
}


# ---------------------------------------------------------------------------
# Per-group registration functions
# ---------------------------------------------------------------------------


def _register_las(mcp: FastMCP) -> None:
    from petro_mcp.tools.las import (
        get_curve_data,
        get_well_header,
        list_curves,
        read_las_file,
    )

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


def _register_production(mcp: FastMCP) -> None:
    from petro_mcp.tools.production import query_production_data
    from petro_mcp.tools.trends import analyze_production_trends

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


def _register_decline(mcp: FastMCP) -> None:
    from petro_mcp.tools.decline import calculate_eur, fit_decline_curve
    from petro_mcp.tools.advanced_decline import (
        fit_duong_decline as _fit_duong,
        fit_ple_decline as _fit_ple,
        fit_sepd_decline as _fit_sepd,
        forecast_advanced_decline as _forecast_advanced,
    )

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

    @mcp.tool()
    def fit_ple_decline(
        production_data: list[dict[str, float]],
    ) -> str:
        """Fit Power Law Exponential (PLE) decline model to production data.

        The PLE model (Ilk et al., 2008) captures transient and boundary-dominated
        flow regimes in tight/shale reservoirs.  Uses petbox-dca for the forward model.

        Args:
            production_data: List of dicts with 'time' (months) and 'rate' keys,
                or 'oil'/'gas' keys (time assumed as sequential months).
        """
        return _fit_ple(production_data)

    @mcp.tool()
    def fit_duong_decline(
        production_data: list[dict[str, float]],
    ) -> str:
        """Fit Duong decline model to production data using petbox-dca.

        The Duong model (2011) is designed for fracture-dominated flow in
        unconventional/shale reservoirs.  Widely used for tight oil and shale gas.

        Args:
            production_data: List of dicts with 'time' (months) and 'rate' keys,
                or 'oil'/'gas' keys (time assumed as sequential months).
        """
        return _fit_duong(production_data)

    @mcp.tool()
    def fit_sepd_decline(
        production_data: list[dict[str, float]],
    ) -> str:
        """Fit Stretched Exponential (SEPD) decline model to production data.

        The SEPD model (Valko, 2009) uses a stretched exponential function effective
        for unconventional reservoirs with heterogeneous fracture networks.

        Args:
            production_data: List of dicts with 'time' (months) and 'rate' keys,
                or 'oil'/'gas' keys (time assumed as sequential months).
        """
        return _fit_sepd(production_data)

    @mcp.tool()
    def forecast_advanced_decline(
        model: str,
        parameters: dict[str, float],
        forecast_months: int = 360,
        economic_limit: float = 5.0,
    ) -> str:
        """Forecast production using an advanced decline model (PLE, Duong, SEPD, THM).

        Generates rate-time forecast and cumulative production using petbox-dca models.
        Use parameters from fit_ple_decline, fit_duong_decline, or fit_sepd_decline,
        or provide THM parameters directly.

        Args:
            model: Model name - 'ple', 'duong', 'sepd', or 'thm'.
            parameters: Dict of model parameters.
                PLE: qi, Di, Dinf, n
                Duong: qi, a, m
                SEPD: qi, tau, n
                THM: qi, Di, bi, bf, telf (optional: bterm, tterm)
            forecast_months: Number of months to forecast (default 360 = 30 years).
            economic_limit: Minimum economic rate in vol/day (default 5.0).
        """
        return _forecast_advanced(model, parameters, forecast_months, economic_limit)


def _register_calculations(mcp: FastMCP) -> None:
    from petro_mcp.tools.calculations import calculate_ip_ratio, nodal_analysis

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


def _register_reservoir(mcp: FastMCP) -> None:
    from petro_mcp.tools.reservoir import (
        calculate_havlena_odeh as _havlena_odeh,
        calculate_pz_analysis as _pz_analysis,
        calculate_radius_of_investigation as _radius_of_investigation,
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
    def havlena_odeh(
        pressures: list[float],
        np_values: list[float],
        rp_values: list[float],
        wp_values: list[float],
        wi_values: list[float],
        bo_values: list[float],
        rs_values: list[float],
        bg_values: list[float],
        bw_values: list[float],
        boi: float,
        rsi: float,
        bgi: float,
        cf: float | None = None,
        swi: float | None = None,
    ) -> str:
        """Oil material balance using Havlena-Odeh straight-line method (1963).

        Identifies drive mechanism (depletion, gas cap, water drive) and estimates
        Original Oil In Place (OOIP). Returns F vs Et plot data for diagnostics.

        Args:
            pressures: Reservoir pressures at each time step (psi).
            np_values: Cumulative oil production at each step (STB).
            rp_values: Cumulative producing GOR at each step (scf/STB).
            wp_values: Cumulative water production at each step (STB).
            wi_values: Cumulative water injection at each step (STB).
            bo_values: Oil FVF at each pressure (bbl/STB).
            rs_values: Solution GOR at each pressure (scf/STB).
            bg_values: Gas FVF at each pressure (bbl/scf).
            bw_values: Water FVF at each pressure (bbl/STB).
            boi: Initial oil FVF (bbl/STB).
            rsi: Initial solution GOR (scf/STB).
            bgi: Initial gas FVF (bbl/scf).
            cf: Formation compressibility (1/psi). Optional.
            swi: Initial water saturation (fraction, 0-1). Optional.
        """
        return _havlena_odeh(
            pressures, np_values, rp_values, wp_values, wi_values,
            bo_values, rs_values, bg_values, bw_values,
            boi, rsi, bgi, cf, swi,
        )

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

    @mcp.tool()
    def radius_of_investigation(
        permeability_md: float,
        time_hours: float,
        porosity: float,
        viscosity_cp: float,
        total_compressibility: float,
    ) -> str:
        """Calculate radius of investigation for a well test.

        r_inv = 0.029 * sqrt(k*t / (phi*mu*ct)), from Lee (1982).

        Args:
            permeability_md: Formation permeability in millidarcies.
            time_hours: Elapsed time in hours.
            porosity: Porosity (fraction, 0-1).
            viscosity_cp: Fluid viscosity in centipoise.
            total_compressibility: Total system compressibility in 1/psi.
        """
        return _radius_of_investigation(
            permeability_md, time_hours, porosity, viscosity_cp, total_compressibility,
        )


def _register_pvt(mcp: FastMCP) -> None:
    from petro_mcp.tools.pvt import (
        bubble_point as _bubble_point,
        calculate_brine_properties as _calculate_brine,
        calculate_gas_z_factor as _calculate_gas_z,
        calculate_oil_compressibility as _calculate_oil_co,
        calculate_pvt as _calculate_pvt,
    )

    @mcp.tool()
    def calculate_pvt_properties(
        api_gravity: float,
        gas_sg: float,
        temperature: float,
        pressure: float,
        separator_pressure: float = 100.0,
        separator_temperature: float = 60.0,
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
            separator_temperature: Separator temperature in F (default 60).
            correlation: Oil correlation set -- 'standing', 'vasquez_beggs',
                or 'petrosky_farshad'.
        """
        return _calculate_pvt(
            api_gravity, gas_sg, temperature, pressure,
            separator_pressure, separator_temperature, correlation,
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
    def calculate_oil_co(
        api_gravity: float,
        gas_sg: float,
        temperature: float,
        pressure: float,
        bubble_point_pressure: float | None = None,
        rs_at_pb: float | None = None,
    ) -> str:
        """Calculate oil compressibility above and below bubble point.

        Uses Vasquez-Beggs (1980) above Pb and material-balance approach below Pb.

        Args:
            api_gravity: Oil API gravity (degrees).
            gas_sg: Gas specific gravity (air = 1.0).
            temperature: Reservoir temperature in F.
            pressure: Current reservoir pressure in psi.
            bubble_point_pressure: Known bubble point pressure in psi (optional).
            rs_at_pb: Solution GOR at bubble point in scf/STB (optional).
        """
        return _calculate_oil_co(
            api_gravity, gas_sg, temperature, pressure,
            bubble_point_pressure, rs_at_pb,
        )

    @mcp.tool()
    def calculate_brine_pvt(
        temperature: float,
        pressure: float,
        salinity: float = 0.0,
    ) -> str:
        """Calculate brine/formation water PVT properties.

        Returns density, viscosity, FVF, and compressibility using
        McCain (1990) and Osif (1988) correlations.

        Args:
            temperature: Formation temperature in F.
            pressure: Formation pressure in psi.
            salinity: Total dissolved solids in ppm (default 0 = fresh water).
        """
        return _calculate_brine(temperature, pressure, salinity)

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
        calculate_effective_porosity as _effective_porosity,
        calculate_hpt as _hpt,
        calculate_indonesian_sw as _indonesian_sw,
        calculate_net_pay as _net_pay,
        calculate_neutron_density_porosity as _nd_porosity,
        calculate_permeability_coates as _perm_coates,
        calculate_permeability_timur as _perm_timur,
        calculate_simandoux_sw as _simandoux_sw,
        calculate_sonic_porosity as _sonic_porosity,
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
    def calculate_sonic_porosity(
        dt: float, dt_matrix: float = 55.5, dt_fluid: float = 189.0, method: str = "wyllie",
    ) -> str:
        """Calculate porosity from sonic (compressional) log.

        Methods: wyllie (time-average), raymer (Raymer-Hunt-Gardner).

        Args:
            dt: Interval transit time (us/ft).
            dt_matrix: Matrix transit time (us/ft). Default 55.5 (sandstone).
            dt_fluid: Fluid transit time (us/ft). Default 189.0.
            method: 'wyllie' or 'raymer'. Default 'wyllie'.
        """
        return _sonic_porosity(dt, dt_matrix, dt_fluid, method)

    @mcp.tool()
    def calculate_nd_porosity(nphi: float, dphi: float) -> str:
        """Quick-look porosity from neutron-density combination (RMS method).

        Args:
            nphi: Neutron porosity (fraction v/v).
            dphi: Density porosity (fraction v/v).
        """
        return _nd_porosity(nphi, dphi)

    @mcp.tool()
    def calculate_effective_porosity(phi_total: float, vshale: float) -> str:
        """Calculate effective porosity from total porosity and shale volume.

        PHIE = PHIT * (1 - Vshale)

        Args:
            phi_total: Total porosity (fraction v/v, 0-1).
            vshale: Shale volume (fraction v/v, 0-1).
        """
        return _effective_porosity(phi_total, vshale)

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
    def calculate_simandoux_sw(
        rt: float, phi: float, rw: float, vshale: float, rsh: float,
        a: float = 1.0, m: float = 2.0, n: float = 2.0,
    ) -> str:
        """Calculate water saturation using Simandoux equation (shaly sands).

        Args:
            rt: True formation resistivity (ohm-m).
            phi: Porosity (fraction v/v, 0-1).
            rw: Formation water resistivity (ohm-m).
            vshale: Shale volume (fraction v/v, 0-1).
            rsh: Shale resistivity (ohm-m).
            a: Tortuosity factor. Default 1.0.
            m: Cementation exponent. Default 2.0.
            n: Saturation exponent. Default 2.0.
        """
        return _simandoux_sw(rt, phi, rw, vshale, rsh, a, m, n)

    @mcp.tool()
    def calculate_indonesian_sw(
        rt: float, phi: float, rw: float, vshale: float, rsh: float,
        a: float = 1.0, m: float = 2.0, n: float = 2.0,
    ) -> str:
        """Calculate water saturation using Indonesian equation (high-Vshale formations).

        Poupon and Leveaux (1971). Better than Archie/Simandoux for very shaly sands.

        Args:
            rt: True formation resistivity (ohm-m).
            phi: Porosity (fraction v/v, 0-1).
            rw: Formation water resistivity (ohm-m).
            vshale: Shale volume (fraction v/v, 0-1).
            rsh: Shale resistivity (ohm-m).
            a: Tortuosity factor. Default 1.0.
            m: Cementation exponent. Default 2.0.
            n: Saturation exponent. Default 2.0.
        """
        return _indonesian_sw(rt, phi, rw, vshale, rsh, a, m, n)

    @mcp.tool()
    def calculate_permeability_timur(phi: float, swirr: float) -> str:
        """Estimate permeability using Timur (1968) equation.

        k = 0.136 * phi^4.4 / Swirr^2

        Args:
            phi: Porosity (fraction v/v, 0-1).
            swirr: Irreducible water saturation (fraction v/v, 0-1).
        """
        return _perm_timur(phi, swirr)

    @mcp.tool()
    def calculate_permeability_coates(
        phi: float, bvi: float, ffi: float, c: float = 10.0,
    ) -> str:
        """Estimate permeability using Coates (1991) equation.

        k = ((phi / C)^2 * (FFI / BVI))^2

        Args:
            phi: Porosity (fraction v/v, 0-1).
            bvi: Bound volume irreducible (fraction v/v).
            ffi: Free fluid index (fraction v/v).
            c: Coates constant. Default 10.0.
        """
        return _perm_coates(phi, bvi, ffi, c)

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

    @mcp.tool()
    def calculate_hpt(
        thickness: float, phi: float, sw: float, ntg: float = 1.0,
    ) -> str:
        """Calculate hydrocarbon pore thickness (HPT = h * phi * (1-Sw) * NTG).

        Args:
            thickness: Net or gross thickness (ft).
            phi: Average porosity (fraction v/v, 0-1).
            sw: Average water saturation (fraction v/v, 0-1).
            ntg: Net-to-gross ratio (0-1). Default 1.0.
        """
        return _hpt(thickness, phi, sw, ntg)


def _register_drilling(mcp: FastMCP) -> None:
    from petro_mcp.tools.drilling import (
        calculate_annular_velocity as _annular_velocity,
        calculate_bit_pressure_drop as _bit_pressure_drop,
        calculate_burst_pressure as _burst_pressure,
        calculate_collapse_pressure as _collapse_pressure,
        calculate_ecd as _ecd,
        calculate_formation_pressure_gradient as _fpg,
        calculate_hydrostatic_pressure as _hydrostatic,
        calculate_icp_fcp as _icp_fcp,
        calculate_kill_mud_weight as _kill_mw,
        calculate_maasp as _maasp,
        calculate_nozzle_tfa as _nozzle_tfa,
    )

    @mcp.tool()
    def calculate_hydrostatic_pressure(mud_weight_ppg: float, tvd_ft: float) -> str:
        """Calculate hydrostatic pressure (P = 0.052 * MW * TVD).

        Args:
            mud_weight_ppg: Mud weight in pounds per gallon (ppg).
            tvd_ft: True vertical depth in feet.
        """
        return _hydrostatic(mud_weight_ppg, tvd_ft)

    @mcp.tool()
    def calculate_ecd(
        mud_weight_ppg: float, annular_pressure_loss_psi: float, tvd_ft: float,
    ) -> str:
        """Calculate equivalent circulating density (ECD = MW + APL / (0.052 * TVD)).

        Args:
            mud_weight_ppg: Static mud weight (ppg).
            annular_pressure_loss_psi: Annular pressure loss (psi).
            tvd_ft: True vertical depth (ft).
        """
        return _ecd(mud_weight_ppg, annular_pressure_loss_psi, tvd_ft)

    @mcp.tool()
    def calculate_formation_pressure_gradient(pressure_psi: float, tvd_ft: float) -> str:
        """Calculate formation pressure gradient as ppg equivalent (FPG = P / (0.052 * TVD)).

        Args:
            pressure_psi: Formation pressure (psi).
            tvd_ft: True vertical depth (ft).
        """
        return _fpg(pressure_psi, tvd_ft)

    @mcp.tool()
    def calculate_kill_mud_weight(
        sidp_psi: float, original_mud_weight_ppg: float, tvd_ft: float,
    ) -> str:
        """Calculate kill mud weight for well control (Kill MW = MW + SIDP / (0.052 * TVD)).

        Args:
            sidp_psi: Shut-in drill pipe pressure (psi).
            original_mud_weight_ppg: Original mud weight (ppg).
            tvd_ft: True vertical depth (ft).
        """
        return _kill_mw(sidp_psi, original_mud_weight_ppg, tvd_ft)

    @mcp.tool()
    def calculate_icp_fcp(
        sidp_psi: float, circulating_pressure_psi: float,
        kill_mud_weight_ppg: float, original_mud_weight_ppg: float,
    ) -> str:
        """Calculate Initial and Final Circulating Pressures for well kill operations.

        ICP = SIDP + slow circulating pressure.
        FCP = SCP * (Kill MW / Original MW).

        Args:
            sidp_psi: Shut-in drill pipe pressure (psi).
            circulating_pressure_psi: Slow circulating pressure (psi).
            kill_mud_weight_ppg: Kill mud weight (ppg).
            original_mud_weight_ppg: Original mud weight (ppg).
        """
        return _icp_fcp(sidp_psi, circulating_pressure_psi, kill_mud_weight_ppg, original_mud_weight_ppg)

    @mcp.tool()
    def calculate_maasp(
        fracture_gradient_ppg: float, mud_weight_ppg: float, shoe_tvd_ft: float,
    ) -> str:
        """Calculate Maximum Allowable Annular Surface Pressure.

        MAASP = (FG - MW) * 0.052 * shoe TVD.

        Args:
            fracture_gradient_ppg: Fracture gradient at shoe (ppg).
            mud_weight_ppg: Current mud weight (ppg).
            shoe_tvd_ft: Casing shoe TVD (ft).
        """
        return _maasp(fracture_gradient_ppg, mud_weight_ppg, shoe_tvd_ft)

    @mcp.tool()
    def calculate_annular_velocity(
        flow_rate_gpm: float, hole_diameter_in: float, pipe_od_in: float,
    ) -> str:
        """Calculate annular velocity (AV = 24.51 * Q / (Dh^2 - Dp^2)).

        Args:
            flow_rate_gpm: Flow rate in gallons per minute.
            hole_diameter_in: Hole or casing ID (inches).
            pipe_od_in: Pipe or drill string OD (inches).
        """
        return _annular_velocity(flow_rate_gpm, hole_diameter_in, pipe_od_in)

    @mcp.tool()
    def calculate_nozzle_tfa(nozzle_sizes: list[int]) -> str:
        """Calculate total flow area (TFA) of bit nozzles.

        TFA = sum(pi/4 * (d/32)^2) for each nozzle size in 32nds of an inch.

        Args:
            nozzle_sizes: List of nozzle sizes in 32nds of an inch (e.g. [12, 12, 12]).
        """
        return _nozzle_tfa(nozzle_sizes)

    @mcp.tool()
    def calculate_bit_pressure_drop(
        flow_rate_gpm: float, mud_weight_ppg: float, tfa_sqin: float,
    ) -> str:
        """Calculate pressure drop across the bit (dP = MW * Q^2 / (12032 * TFA^2)).

        Args:
            flow_rate_gpm: Flow rate in gallons per minute.
            mud_weight_ppg: Mud weight (ppg).
            tfa_sqin: Total flow area of nozzles (in^2).
        """
        return _bit_pressure_drop(flow_rate_gpm, mud_weight_ppg, tfa_sqin)

    @mcp.tool()
    def calculate_burst_pressure(
        yield_strength_psi: float, wall_thickness_in: float, od_in: float,
    ) -> str:
        """Calculate internal burst pressure using Barlow's formula with API 12.5% tolerance.

        P_burst = 0.875 * 2 * Fy * t / OD.

        Args:
            yield_strength_psi: Minimum yield strength (psi).
            wall_thickness_in: Nominal wall thickness (inches).
            od_in: Outer diameter (inches).
        """
        return _burst_pressure(yield_strength_psi, wall_thickness_in, od_in)

    @mcp.tool()
    def calculate_collapse_pressure(
        od_in: float, wall_thickness_in: float, yield_strength_psi: float, grade: str = "",
    ) -> str:
        """Calculate collapse pressure rating per API 5C3.

        Determines regime (yield, plastic, transition, elastic) from D/t ratio
        and yield strength, then applies the corresponding API formula.

        Args:
            od_in: Casing outer diameter (inches).
            wall_thickness_in: Wall thickness (inches).
            yield_strength_psi: Minimum yield strength (psi).
            grade: Optional API grade label (e.g. 'N-80') for reference.
        """
        return _collapse_pressure(od_in, wall_thickness_in, yield_strength_psi, grade)


def _register_economics(mcp: FastMCP) -> None:
    from petro_mcp.tools.economics import (
        calculate_breakeven_price as _breakeven_price,
        calculate_irr as _irr,
        calculate_npv as _npv,
        calculate_operating_netback as _operating_netback,
        calculate_payout_period as _payout_period,
        calculate_price_sensitivity as _price_sensitivity,
        calculate_pv10 as _pv10,
        calculate_well_economics as _well_economics,
    )

    @mcp.tool()
    def calculate_well_economics(
        monthly_oil_bbl: list[float],
        monthly_gas_mcf: list[float],
        monthly_water_bbl: list[float],
        oil_price_bbl: float,
        gas_price_mcf: float,
        opex_monthly: float,
        capex: float,
        royalty_pct: float = 0.125,
        tax_rate: float = 0.0,
        discount_rate: float = 0.10,
        working_interest: float = 1.0,
        net_revenue_interest: float = 0.875,
    ) -> str:
        """Full discounted cash flow analysis for a well.

        Takes production arrays (from decline forecast) plus economic assumptions.
        Returns NPV, IRR, payout period, profitability index, and monthly cash flows.

        Args:
            monthly_oil_bbl: Monthly oil production (bbl) for each period.
            monthly_gas_mcf: Monthly gas production (Mcf) for each period.
            monthly_water_bbl: Monthly water production (bbl) for each period.
            oil_price_bbl: Oil price ($/bbl).
            gas_price_mcf: Gas price ($/Mcf).
            opex_monthly: Monthly operating expense ($).
            capex: Total capital expenditure ($), applied at time 0.
            royalty_pct: Royalty fraction (0-1). Default 0.125.
            tax_rate: Severance/production tax rate (0-1). Default 0.0.
            discount_rate: Annual discount rate for NPV. Default 0.10.
            working_interest: Working interest fraction (0-1). Default 1.0.
            net_revenue_interest: Net revenue interest fraction (0-1). Default 0.875.
        """
        return _well_economics(
            monthly_oil_bbl, monthly_gas_mcf, monthly_water_bbl,
            oil_price_bbl, gas_price_mcf, opex_monthly, capex,
            royalty_pct, tax_rate, discount_rate, working_interest, net_revenue_interest,
        )

    @mcp.tool()
    def calculate_npv(
        cash_flows: list[float],
        discount_rate: float = 0.10,
    ) -> str:
        """Calculate Net Present Value from monthly cash flows.

        NPV = sum(CF_t / (1 + r/12)^t) for t = 0, 1, 2, ...

        Args:
            cash_flows: Monthly cash flows ($). First element is typically negative (capex).
            discount_rate: Annual discount rate. Default 0.10.
        """
        return _npv(cash_flows, discount_rate)

    @mcp.tool()
    def calculate_irr(
        cash_flows: list[float],
    ) -> str:
        """Calculate Internal Rate of Return via bisection.

        IRR is the annual discount rate at which NPV = 0.

        Args:
            cash_flows: Monthly cash flows ($). First element is typically negative (capex).
        """
        return _irr(cash_flows)

    @mcp.tool()
    def calculate_pv10(
        monthly_net_revenue: list[float],
    ) -> str:
        """Calculate PV10 -- SEC standard present value at 10% annual discount.

        PV10 = sum(NR_t / 1.10^(t/12))

        Args:
            monthly_net_revenue: Monthly net revenue ($) after royalties and opex.
        """
        return _pv10(monthly_net_revenue)

    @mcp.tool()
    def calculate_breakeven_price(
        monthly_production_bbl: list[float],
        monthly_opex: float,
        capex: float,
        discount_rate: float = 0.10,
        royalty_pct: float = 0.125,
        months: int | None = None,
    ) -> str:
        """Calculate breakeven oil price -- minimum price for NPV = 0.

        Uses bisection to find the oil price at which discounted net cash flow equals zero.

        Args:
            monthly_production_bbl: Monthly oil production (bbl) per period.
            monthly_opex: Monthly operating expense ($).
            capex: Total capital expenditure ($).
            discount_rate: Annual discount rate. Default 0.10.
            royalty_pct: Royalty fraction (0-1). Default 0.125.
            months: Number of months to evaluate (default: length of production array).
        """
        return _breakeven_price(monthly_production_bbl, monthly_opex, capex,
                                discount_rate, royalty_pct, months)

    @mcp.tool()
    def calculate_operating_netback(
        oil_price: float,
        gas_price: float,
        oil_rate_bpd: float,
        gas_rate_mcfd: float,
        opex_per_boe: float,
        royalty_pct: float = 0.125,
        transport_per_boe: float = 0.0,
    ) -> str:
        """Calculate operating netback per BOE.

        Revenue - royalties - opex - transport per BOE. Gas at 6 Mcf/BOE.

        Args:
            oil_price: Oil price ($/bbl).
            gas_price: Gas price ($/Mcf).
            oil_rate_bpd: Oil production rate (bbl/day).
            gas_rate_mcfd: Gas production rate (Mcf/day).
            opex_per_boe: Operating expense per BOE ($/BOE).
            royalty_pct: Royalty fraction (0-1). Default 0.125.
            transport_per_boe: Transportation cost per BOE ($/BOE). Default 0.0.
        """
        return _operating_netback(oil_price, gas_price, oil_rate_bpd, gas_rate_mcfd,
                                  opex_per_boe, royalty_pct, transport_per_boe)

    @mcp.tool()
    def calculate_payout_period(
        cash_flows: list[float],
    ) -> str:
        """Calculate payout period -- months to recover initial investment.

        Payout is the first month where cumulative cash flow >= 0.

        Args:
            cash_flows: Monthly cash flows ($). First element is typically negative (capex).
        """
        return _payout_period(cash_flows)

    @mcp.tool()
    def calculate_price_sensitivity(
        monthly_oil_bbl: list[float],
        monthly_gas_mcf: list[float],
        monthly_water_bbl: list[float],
        opex_monthly: float,
        capex: float,
        price_scenarios: list[dict[str, float]],
        discount_rate: float = 0.10,
        royalty_pct: float = 0.125,
    ) -> str:
        """Calculate NPV across multiple price scenarios for sensitivity/tornado charts.

        Args:
            monthly_oil_bbl: Monthly oil production (bbl) per period.
            monthly_gas_mcf: Monthly gas production (Mcf) per period.
            monthly_water_bbl: Monthly water production (bbl) per period.
            opex_monthly: Monthly operating expense ($).
            capex: Total capital expenditure ($).
            price_scenarios: List of dicts with 'oil_price' and 'gas_price'.
            discount_rate: Annual discount rate. Default 0.10.
            royalty_pct: Royalty fraction (0-1). Default 0.125.
        """
        return _price_sensitivity(monthly_oil_bbl, monthly_gas_mcf, monthly_water_bbl,
                                  opex_monthly, capex, price_scenarios,
                                  discount_rate, royalty_pct)


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


def _register_production_eng(mcp: FastMCP) -> None:
    from petro_mcp.tools.production_eng import (
        calculate_beggs_brill_pressure_drop as _beggs_brill,
        calculate_coleman_critical_rate as _coleman_critical,
        calculate_critical_choke_flow as _choke_flow,
        calculate_erosional_velocity as _erosional_velocity,
        calculate_hydrate_inhibitor_dosing as _hydrate_inhibitor,
        calculate_hydrate_temperature as _hydrate_temp,
        calculate_turner_critical_rate as _turner_critical,
    )

    @mcp.tool()
    def calculate_beggs_brill(
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
        """Beggs & Brill (1973) multiphase pressure drop in pipes.

        The most widely used multiphase flow correlation. Determines flow pattern,
        calculates liquid holdup, friction factor, and pressure gradient including
        elevation, friction, and acceleration terms.

        Args:
            flow_rate_bpd: Total liquid flow rate in bbl/day.
            gor_scf_bbl: Gas-oil ratio in scf/bbl.
            water_cut: Water cut as fraction (0-1).
            oil_api: Oil API gravity.
            gas_sg: Gas specific gravity (air = 1.0).
            pipe_id_in: Pipe inner diameter in inches.
            pipe_length_ft: Pipe length in feet.
            inclination_deg: Pipe inclination from horizontal (-90 to 90 degrees).
            wellhead_pressure_psi: Wellhead (outlet) pressure in psi.
            temperature_f: Average flowing temperature in degrees F.
        """
        return _beggs_brill(
            flow_rate_bpd, gor_scf_bbl, water_cut, oil_api, gas_sg,
            pipe_id_in, pipe_length_ft, inclination_deg,
            wellhead_pressure_psi, temperature_f,
        )

    @mcp.tool()
    def calculate_turner_critical(
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
        lift liquids from a gas well using the droplet model.

        Args:
            wellhead_pressure_psi: Wellhead flowing pressure in psi.
            wellhead_temp_f: Wellhead temperature in degrees F.
            gas_sg: Gas specific gravity (air = 1.0).
            condensate_sg: Condensate specific gravity (optional).
            water_sg: Water specific gravity. Default 1.07.
            tubing_id_in: Tubing inner diameter in inches. Default 2.441.
            current_rate_mcfd: Current gas rate in Mcf/d for status check (optional).
        """
        return _turner_critical(
            wellhead_pressure_psi, wellhead_temp_f, gas_sg,
            condensate_sg, water_sg, tubing_id_in, current_rate_mcfd,
        )

    @mcp.tool()
    def calculate_coleman_critical(
        wellhead_pressure_psi: float,
        wellhead_temp_f: float,
        gas_sg: float,
        tubing_id_in: float = 2.441,
        current_rate_mcfd: float | None = None,
    ) -> str:
        """Coleman et al. (1991) critical rate for liquid loading (20% below Turner).

        Recommended for low-pressure gas wells (< ~500 psi wellhead pressure).

        Args:
            wellhead_pressure_psi: Wellhead flowing pressure in psi.
            wellhead_temp_f: Wellhead temperature in degrees F.
            gas_sg: Gas specific gravity (air = 1.0).
            tubing_id_in: Tubing inner diameter in inches. Default 2.441.
            current_rate_mcfd: Current gas rate in Mcf/d for status check (optional).
        """
        return _coleman_critical(
            wellhead_pressure_psi, wellhead_temp_f, gas_sg,
            tubing_id_in, current_rate_mcfd,
        )

    @mcp.tool()
    def calculate_hydrate_temp(
        pressure_psi: float,
        gas_sg: float,
    ) -> str:
        """Estimate hydrate formation temperature using gas-gravity method (Katz chart).

        Args:
            pressure_psi: System pressure in psi.
            gas_sg: Gas specific gravity (air = 1.0, range 0.55-1.0).
        """
        return _hydrate_temp(pressure_psi, gas_sg)

    @mcp.tool()
    def calculate_hydrate_inhibitor(
        hydrate_temp_f: float,
        operating_temp_f: float,
        water_rate_bwpd: float,
        inhibitor: str = "methanol",
    ) -> str:
        """Calculate hydrate inhibitor injection rate using Hammerschmidt equation.

        Supports methanol, MEG, and ethanol.

        Args:
            hydrate_temp_f: Hydrate formation temperature in degrees F.
            operating_temp_f: Target operating temperature in degrees F.
            water_rate_bwpd: Water production rate in bbl/day.
            inhibitor: Inhibitor type - 'methanol', 'meg', or 'ethanol'.
        """
        return _hydrate_inhibitor(hydrate_temp_f, operating_temp_f, water_rate_bwpd, inhibitor)

    @mcp.tool()
    def calculate_erosional_vel(
        density_mix_lb_ft3: float,
        c_factor: float = 100.0,
    ) -> str:
        """Calculate erosional velocity per API RP 14E (v_e = C / sqrt(rho_mix)).

        Args:
            density_mix_lb_ft3: Mixture density in lb/ft3.
            c_factor: Erosional constant. Default 100. Use 125 for intermittent,
                150-200 for corrosion-resistant alloys.
        """
        return _erosional_velocity(density_mix_lb_ft3, c_factor)

    @mcp.tool()
    def calculate_choke_flow(
        upstream_pressure_psi: float,
        choke_size_64ths: float,
        gor_scf_bbl: float,
        oil_api: float,
        water_cut: float = 0.0,
        gas_sg: float = 0.65,
    ) -> str:
        """Calculate flow rate through a choke using Gilbert correlation (1954).

        q = P * S^1.89 / (435 * GLR^0.546). Valid for critical (sonic) flow only.

        Args:
            upstream_pressure_psi: Upstream pressure in psi.
            choke_size_64ths: Choke bean size in 64ths of an inch.
            gor_scf_bbl: Gas-oil ratio in scf/bbl.
            oil_api: Oil API gravity.
            water_cut: Water cut as fraction (0-1). Default 0.0.
            gas_sg: Gas specific gravity. Default 0.65.
        """
        return _choke_flow(
            upstream_pressure_psi, choke_size_64ths, gor_scf_bbl,
            oil_api, water_cut, gas_sg,
        )


def _register_trajectory(mcp: FastMCP) -> None:
    try:
        from petro_mcp.tools.trajectory import (
            calculate_survey as _calculate_survey,
            calculate_dogleg_severity as _calculate_dls,
            calculate_vertical_section as _calculate_vs,
            calculate_tortuosity as _calculate_tortuosity,
            check_anticollision as _check_anticollision,
        )
    except ImportError:
        return  # welleng not installed, skip trajectory tools

    @mcp.tool()
    def calculate_well_survey(
        md: list[float],
        inclination: list[float],
        azimuth: list[float],
        unit: str = "feet",
    ) -> str:
        """Calculate well trajectory using minimum curvature method.

        Takes survey station data (MD, inclination, azimuth) and returns
        computed North, East, TVD, and dogleg severity at each station.

        Args:
            md: List of measured depths (ft or m).
            inclination: List of inclinations (degrees from vertical, 0-180).
            azimuth: List of azimuths (degrees from north, 0-360).
            unit: Depth unit -- 'feet' or 'meters'. Default 'feet'.
        """
        return _calculate_survey(md, inclination, azimuth, unit)

    @mcp.tool()
    def calculate_dogleg_severity(
        md1: float,
        inc1: float,
        azi1: float,
        md2: float,
        inc2: float,
        azi2: float,
        course_length_unit: str = "feet",
    ) -> str:
        """Calculate dogleg severity between two survey stations.

        Returns DLS in deg/100ft (or deg/30m for metric).

        Args:
            md1: Measured depth at station 1.
            inc1: Inclination at station 1 (degrees).
            azi1: Azimuth at station 1 (degrees).
            md2: Measured depth at station 2.
            inc2: Inclination at station 2 (degrees).
            azi2: Azimuth at station 2 (degrees).
            course_length_unit: 'feet' or 'meters'. Default 'feet'.
        """
        return _calculate_dls(md1, inc1, azi1, md2, inc2, azi2, course_length_unit)

    @mcp.tool()
    def calculate_vertical_section(
        md: list[float],
        inclination: list[float],
        azimuth: list[float],
        vs_azimuth: float = 0.0,
        unit: str = "feet",
    ) -> str:
        """Project well trajectory onto a vertical section plane.

        Calculates the horizontal displacement projected onto a plane at the
        given azimuth. Standard way to view a well path in 2D cross-section.

        Args:
            md: List of measured depths.
            inclination: List of inclinations (degrees).
            azimuth: List of azimuths (degrees).
            vs_azimuth: Vertical section azimuth in degrees (0 = North). Default 0.
            unit: Depth unit -- 'feet' or 'meters'. Default 'feet'.
        """
        return _calculate_vs(md, inclination, azimuth, vs_azimuth, unit)

    @mcp.tool()
    def calculate_wellbore_tortuosity(
        md: list[float],
        inclination: list[float],
        azimuth: list[float],
        unit: str = "feet",
    ) -> str:
        """Calculate wellbore tortuosity index from survey data.

        Tortuosity measures how much the wellbore deviates from an ideal path.
        Higher values indicate more tortuous wellpath, impacting drilling and
        production operations.

        Args:
            md: List of measured depths.
            inclination: List of inclinations (degrees).
            azimuth: List of azimuths (degrees).
            unit: Depth unit -- 'feet' or 'meters'. Default 'feet'.
        """
        return _calculate_tortuosity(md, inclination, azimuth, unit)

    @mcp.tool()
    def check_well_anticollision(
        well1_md: list[float],
        well1_inc: list[float],
        well1_azi: list[float],
        well2_md: list[float],
        well2_inc: list[float],
        well2_azi: list[float],
        well2_start_north: float = 0.0,
        well2_start_east: float = 0.0,
        unit: str = "feet",
    ) -> str:
        """Check separation between two wells at closest approach.

        Computes center-to-center distance between two well trajectories
        and identifies the closest approach point.

        Args:
            well1_md: Measured depths for reference well.
            well1_inc: Inclinations for reference well (degrees).
            well1_azi: Azimuths for reference well (degrees).
            well2_md: Measured depths for offset well.
            well2_inc: Inclinations for offset well (degrees).
            well2_azi: Azimuths for offset well (degrees).
            well2_start_north: Offset well surface location north of reference.
            well2_start_east: Offset well surface location east of reference.
            unit: Depth unit -- 'feet' or 'meters'. Default 'feet'.
        """
        return _check_anticollision(
            well1_md, well1_inc, well1_azi,
            well2_md, well2_inc, well2_azi,
            well2_start_north, well2_start_east, unit,
        )


def _register_resources(mcp: FastMCP) -> None:
    @mcp.resource("wells://list/{directory}")
    def browse_wells(directory: str) -> str:
        """Browse wells in a directory of LAS and CSV files."""
        return list_wells(directory)

    @mcp.resource("wells://production/{file_path}")
    def get_production_summary(file_path: str) -> str:
        """Get production summary for all wells in a CSV file."""
        return production_summary(file_path)


def _register_prompts(mcp: FastMCP) -> None:
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


# ---------------------------------------------------------------------------
# Mapping from group name -> registration function
# ---------------------------------------------------------------------------

_GROUP_REGISTRARS: dict[str, callable] = {
    "las": _register_las,
    "production": _register_production,
    "decline": _register_decline,
    "calculations": _register_calculations,
    "reservoir": _register_reservoir,
    "pvt": _register_pvt,
    "petrophysics": _register_petrophysics,
    "drilling": _register_drilling,
    "economics": _register_economics,
    "units": _register_units,
    "production_eng": _register_production_eng,
    "trajectory": _register_trajectory,
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
        instructions="Petroleum engineering data and tools for LLMs",
    )

    if groups is None:
        groups = set(TOOL_GROUPS.keys())

    for name in groups:
        registrar = _GROUP_REGISTRARS.get(name)
        if registrar is not None:
            registrar(mcp)

    # Resources and prompts are always registered
    _register_resources(mcp)
    _register_prompts(mcp)

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
