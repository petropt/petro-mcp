"""FastAPI REST service exposing petro-mcp calculation engines.

Powers tools.petropt.com -- the web frontend calls these endpoints
instead of re-implementing formulas in JavaScript.

Run with:
    petro-api            # entry-point (after pip install)
    uvicorn petro_mcp.api.app:app --reload   # dev mode
"""

from __future__ import annotations

import json
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

from petro_mcp.api.rate_limiter import RapidAPIMiddleware

# ---------------------------------------------------------------------------
# Tag metadata for OpenAPI grouping
# ---------------------------------------------------------------------------

tags_metadata = [
    {
        "name": "DCA",
        "description": "Decline Curve Analysis — Arps exponential, hyperbolic, harmonic, modified hyperbolic, and Duong models. Fit production data, forecast rates, and calculate EUR.",
    },
    {
        "name": "PVT",
        "description": "Black-oil PVT correlations — Standing, Vasquez-Beggs, Petrosky-Farshad. Bubble point, solution GOR, FVF, viscosity, compressibility, and gas Z-factor.",
    },
    {
        "name": "Petrophysics",
        "description": "Log analysis calculations — Archie water saturation, density porosity, shale volume (linear, Larionov, Clavier).",
    },
    {
        "name": "Drilling",
        "description": "Drilling engineering — hydrostatic pressure, ECD, kill sheet (kill mud weight, ICP, FCP).",
    },
    {
        "name": "Economics",
        "description": "Petroleum economics — NPV, IRR, payout, full discounted cash flow analysis with royalties, taxes, and working interest.",
    },
    {
        "name": "System",
        "description": "Health checks, API catalog, and service metadata.",
    },
]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Petroleum Engineering Calculator API",
    description=(
        "Production-grade petroleum engineering calculations exposed as a REST API. "
        "Covers decline curve analysis (DCA), PVT correlations, petrophysics, "
        "drilling engineering, and economics. Built by Groundwork Analytics."
    ),
    version="0.7.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Groundwork Analytics",
        "url": "https://petropt.com",
        "email": "info@petropt.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# --- Middleware stack (order matters: last added = first executed) ----------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tools.petropt.com",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(RapidAPIMiddleware)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call(func, **kwargs) -> dict[str, Any]:
    """Call a petro-mcp tool function and return parsed JSON.

    Tool functions return JSON strings. This helper parses the result and
    converts tool-level ValueErrors into 400 responses.
    """
    try:
        result_json = func(**kwargs)
        return json.loads(result_json)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ===================================================================
# DCA endpoints
# ===================================================================

class FitDeclineRequest(BaseModel):
    production_data: list[dict[str, float]] = Field(
        ...,
        description="List of dicts with 'time' and 'rate' keys. Time in months, rate in bbl/day or Mcf/day.",
        json_schema_extra={
            "example": [
                {"time": 0, "rate": 1000},
                {"time": 1, "rate": 900},
                {"time": 2, "rate": 820},
                {"time": 3, "rate": 750},
            ]
        },
    )
    model: str = Field(
        "hyperbolic",
        description="Decline model: exponential, hyperbolic, harmonic, modified_hyperbolic, or duong.",
    )

    model_config = {"json_schema_extra": {"example": {
        "production_data": [
            {"time": 0, "rate": 1000}, {"time": 1, "rate": 900},
            {"time": 2, "rate": 820}, {"time": 3, "rate": 750},
            {"time": 4, "rate": 690}, {"time": 5, "rate": 640},
        ],
        "model": "hyperbolic",
    }}}


class EURRequest(BaseModel):
    qi: float = Field(..., description="Initial production rate (bbl/day or Mcf/day)")
    Di: float = Field(0.0, description="Initial decline rate (1/month)")
    b: float = Field(0.0, description="Arps b-factor (0=exponential, 0<b<1=hyperbolic, 1=harmonic)")
    economic_limit: float = Field(5.0, description="Minimum economic rate (same units as qi)")
    model: str = Field("hyperbolic", description="Decline model name")
    max_time: float = Field(600, description="Max forecast horizon in months")
    Dmin: float = Field(0.005, description="Terminal decline rate for modified_hyperbolic (1/month)")
    a: float = Field(1.0, description="Duong intercept parameter")
    m: float = Field(1.1, description="Duong slope parameter")

    model_config = {"json_schema_extra": {"example": {
        "qi": 800, "Di": 0.06, "b": 1.2,
        "model": "hyperbolic", "economic_limit": 5.0,
    }}}


class ForecastRequest(BaseModel):
    qi: float = Field(..., description="Initial production rate (bbl/day or Mcf/day)")
    Di: float = Field(0.0, description="Initial decline rate (1/month)")
    b: float = Field(0.0, description="Arps b-factor")
    model: str = Field("hyperbolic", description="Decline model name")
    months: int = Field(360, description="Number of months to forecast")
    Dmin: float = Field(0.005, description="Terminal decline for modified_hyperbolic (1/month)")
    a: float = Field(1.0, description="Duong intercept parameter")
    m: float = Field(1.1, description="Duong slope parameter")

    model_config = {"json_schema_extra": {"example": {
        "qi": 500, "Di": 0.05, "b": 1.0,
        "model": "hyperbolic", "months": 60,
    }}}


@app.post(
    "/api/v1/decline/fit",
    tags=["DCA"],
    summary="Fit decline curve to production data",
    description=(
        "Fit an Arps decline curve model to time-rate production data using "
        "non-linear least squares (scipy curve_fit). Returns fitted parameters, "
        "R-squared, and predicted rates. Supports exponential, hyperbolic, "
        "harmonic, modified hyperbolic, and Duong models."
    ),
    response_description="Fitted model parameters, R-squared, and predicted rates.",
)
def fit_decline(req: FitDeclineRequest):
    """Fit an Arps decline curve to production data."""
    from petro_mcp.tools.decline import fit_decline_curve
    return _call(fit_decline_curve, production_data=req.production_data, model=req.model)


@app.post(
    "/api/v1/decline/eur",
    tags=["DCA"],
    summary="Calculate Estimated Ultimate Recovery",
    description=(
        "Calculate EUR (Estimated Ultimate Recovery) by integrating the decline "
        "curve from time zero to the economic limit or max time. Uses trapezoidal "
        "integration for accuracy."
    ),
    response_description="EUR value, cumulative production profile, and economic life.",
)
def decline_eur(req: EURRequest):
    """Calculate Estimated Ultimate Recovery from decline parameters."""
    from petro_mcp.tools.decline import calculate_eur
    return _call(
        calculate_eur,
        qi=req.qi, Di=req.Di, b=req.b,
        economic_limit=req.economic_limit, model=req.model,
        max_time=req.max_time, Dmin=req.Dmin, a=req.a, m=req.m,
    )


@app.post(
    "/api/v1/decline/forecast",
    tags=["DCA"],
    summary="Generate production forecast",
    description=(
        "Generate a month-by-month production rate forecast from given decline "
        "parameters. Returns arrays of months and corresponding rates."
    ),
    response_description="Monthly forecast with time and rate arrays.",
)
def decline_forecast(req: ForecastRequest):
    """Generate a production forecast from decline parameters."""
    import numpy as np
    from petro_mcp.tools.decline import _MODELS

    model = req.model
    if model not in _MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}. Must be one of: {list(_MODELS.keys())}")

    func, param_names, _, _ = _MODELS[model]
    available = {"qi": req.qi, "Di": req.Di, "b": req.b, "Dmin": req.Dmin, "a": req.a, "m": req.m}
    params = [available[p] for p in param_names]

    t = np.arange(0, req.months + 1, dtype=float)
    rates = func(t, *params)
    rates = np.maximum(rates, 0.0)

    return {
        "model": model,
        "parameters": {p: available[p] for p in param_names},
        "months": list(range(req.months + 1)),
        "rates": [round(float(r), 2) for r in rates],
    }


# ===================================================================
# PVT endpoints
# ===================================================================

class PVTRequest(BaseModel):
    api_gravity: float = Field(..., description="Oil API gravity (typically 10-70)")
    gas_sg: float = Field(..., description="Gas specific gravity (air=1.0, typically 0.55-1.2)")
    temperature: float = Field(..., description="Reservoir temperature (deg F)")
    pressure: float = Field(..., description="Current pressure (psi)")
    separator_pressure: float = Field(100.0, description="Separator pressure (psi)")
    separator_temperature: float = Field(60.0, description="Separator temperature (deg F)")
    correlation: str = Field("standing", description="PVT correlation: standing, vasquez_beggs, or petrosky_farshad")

    model_config = {"json_schema_extra": {"example": {
        "api_gravity": 35.0, "gas_sg": 0.75,
        "temperature": 200.0, "pressure": 3000.0,
        "correlation": "standing",
    }}}


class BubblePointRequest(BaseModel):
    api_gravity: float = Field(..., description="Oil API gravity")
    gas_sg: float = Field(..., description="Gas specific gravity")
    temperature: float = Field(..., description="Temperature (deg F)")
    rs: float = Field(..., description="Solution GOR (scf/STB)")

    model_config = {"json_schema_extra": {"example": {
        "api_gravity": 35.0, "gas_sg": 0.75,
        "temperature": 200.0, "rs": 500.0,
    }}}


class GasZRequest(BaseModel):
    temperature: float = Field(..., description="Temperature (deg F)")
    pressure: float = Field(..., description="Pressure (psi)")
    gas_sg: float = Field(..., description="Gas specific gravity (air=1.0)")
    method: str = Field("hall_yarborough", description="Z-factor method: hall_yarborough or dranchuk_abou_kassem")
    pseudocritical_method: str = Field("sutton", description="Pseudocritical method: sutton or piper")
    h2s_fraction: float = Field(0.0, description="Mole fraction of H2S (0-1)")
    co2_fraction: float = Field(0.0, description="Mole fraction of CO2 (0-1)")
    n2_fraction: float = Field(0.0, description="Mole fraction of N2 (0-1)")

    model_config = {"json_schema_extra": {"example": {
        "temperature": 200.0, "pressure": 2000.0, "gas_sg": 0.70,
    }}}


@app.post(
    "/api/v1/pvt/properties",
    tags=["PVT"],
    summary="Calculate black-oil PVT properties",
    description=(
        "Calculate comprehensive black-oil PVT properties including bubble point "
        "pressure, solution GOR, oil FVF, oil viscosity, oil compressibility, and "
        "oil density. Supports Standing, Vasquez-Beggs, and Petrosky-Farshad correlations."
    ),
    response_description="Full PVT property suite for the given conditions.",
)
def pvt_properties(req: PVTRequest):
    """Calculate comprehensive black-oil PVT properties."""
    from petro_mcp.tools.pvt import calculate_pvt
    return _call(
        calculate_pvt,
        api_gravity=req.api_gravity, gas_sg=req.gas_sg,
        temperature=req.temperature, pressure=req.pressure,
        separator_pressure=req.separator_pressure,
        separator_temperature=req.separator_temperature,
        correlation=req.correlation,
    )


@app.post(
    "/api/v1/pvt/bubble-point",
    tags=["PVT"],
    summary="Calculate bubble point pressure",
    description=(
        "Calculate bubble point pressure using the Standing correlation. "
        "Input the API gravity, gas specific gravity, temperature, and solution GOR."
    ),
    response_description="Bubble point pressure in psi.",
)
def pvt_bubble_point(req: BubblePointRequest):
    """Calculate bubble point pressure (Standing)."""
    from petro_mcp.tools.pvt import bubble_point
    return _call(
        bubble_point,
        api_gravity=req.api_gravity, gas_sg=req.gas_sg,
        temperature=req.temperature, rs=req.rs,
    )


@app.post(
    "/api/v1/pvt/z-factor",
    tags=["PVT"],
    summary="Calculate gas Z-factor",
    description=(
        "Calculate the gas compressibility factor (Z-factor) using Hall-Yarborough "
        "or Dranchuk-Abou-Kassem correlations. Supports Wichert-Aziz acid gas "
        "correction for H2S and CO2."
    ),
    response_description="Z-factor and pseudocritical properties.",
)
def pvt_z_factor(req: GasZRequest):
    """Calculate gas Z-factor."""
    from petro_mcp.tools.pvt import calculate_gas_z_factor
    return _call(
        calculate_gas_z_factor,
        temperature=req.temperature, pressure=req.pressure,
        gas_sg=req.gas_sg, method=req.method,
        pseudocritical_method=req.pseudocritical_method,
        h2s_fraction=req.h2s_fraction,
        co2_fraction=req.co2_fraction,
        n2_fraction=req.n2_fraction,
    )


# ===================================================================
# Petrophysics endpoints
# ===================================================================

class ArchieRequest(BaseModel):
    rt: float = Field(..., description="True formation resistivity (ohm-m)")
    phi: float = Field(..., description="Porosity (fraction 0-1)")
    rw: float = Field(..., description="Formation water resistivity (ohm-m)")
    a: float = Field(1.0, description="Tortuosity factor (default 1.0 for carbonates, 0.81 for sandstones)")
    m: float = Field(2.0, description="Cementation exponent (typically 1.8-2.5)")
    n: float = Field(2.0, description="Saturation exponent (typically 1.5-2.5)")

    model_config = {"json_schema_extra": {"example": {
        "rt": 20.0, "phi": 0.20, "rw": 0.05,
    }}}


class DensityPorosityRequest(BaseModel):
    rhob: float = Field(..., description="Bulk density from density log (g/cc)")
    rho_matrix: float = Field(2.65, description="Matrix density (g/cc). 2.65 for sandstone, 2.71 for limestone, 2.87 for dolomite.")
    rho_fluid: float = Field(1.0, description="Fluid density (g/cc)")

    model_config = {"json_schema_extra": {"example": {
        "rhob": 2.40, "rho_matrix": 2.65, "rho_fluid": 1.0,
    }}}


class VshaleRequest(BaseModel):
    gr: float = Field(..., description="Gamma ray log reading at depth of interest (API units)")
    gr_clean: float = Field(..., description="Gamma ray reading in clean sand zone (API units)")
    gr_shale: float = Field(..., description="Gamma ray reading in pure shale zone (API units)")
    method: str = Field("linear", description="Vshale method: linear, larionov_tertiary, larionov_older, or clavier")

    model_config = {"json_schema_extra": {"example": {
        "gr": 70.0, "gr_clean": 20.0, "gr_shale": 120.0, "method": "linear",
    }}}


@app.post(
    "/api/v1/petrophys/archie",
    tags=["Petrophysics"],
    summary="Calculate water saturation (Archie)",
    description=(
        "Calculate water saturation using the Archie equation: "
        "Sw = ((a * Rw) / (phi^m * Rt))^(1/n). Standard parameters for "
        "clean, non-shaly formations."
    ),
    response_description="Water saturation (Sw) as a fraction 0-1.",
)
def petrophys_archie(req: ArchieRequest):
    """Calculate water saturation using Archie equation."""
    from petro_mcp.tools.petrophysics import calculate_archie_sw
    return _call(
        calculate_archie_sw,
        rt=req.rt, phi=req.phi, rw=req.rw,
        a=req.a, m=req.m, n=req.n,
    )


@app.post(
    "/api/v1/petrophys/porosity",
    tags=["Petrophysics"],
    summary="Calculate density porosity",
    description=(
        "Calculate porosity from bulk density log using: "
        "phi = (rho_matrix - rhob) / (rho_matrix - rho_fluid). "
        "Common matrix densities: sandstone=2.65, limestone=2.71, dolomite=2.87 g/cc."
    ),
    response_description="Porosity as a fraction 0-1.",
)
def petrophys_porosity(req: DensityPorosityRequest):
    """Calculate porosity from bulk density."""
    from petro_mcp.tools.petrophysics import calculate_density_porosity
    return _call(
        calculate_density_porosity,
        rhob=req.rhob, rho_matrix=req.rho_matrix, rho_fluid=req.rho_fluid,
    )


@app.post(
    "/api/v1/petrophys/vshale",
    tags=["Petrophysics"],
    summary="Calculate shale volume from gamma ray",
    description=(
        "Calculate shale volume (Vshale) from gamma ray log data. Supports "
        "linear, Larionov (Tertiary and Older rocks), and Clavier methods."
    ),
    response_description="Shale volume (Vshale) as a fraction 0-1.",
)
def petrophys_vshale(req: VshaleRequest):
    """Calculate shale volume from gamma ray."""
    from petro_mcp.tools.petrophysics import calculate_vshale
    return _call(
        calculate_vshale,
        gr=req.gr, gr_clean=req.gr_clean, gr_shale=req.gr_shale,
        method=req.method,
    )


# ===================================================================
# Drilling endpoints
# ===================================================================

class HydrostaticRequest(BaseModel):
    mud_weight_ppg: float = Field(..., description="Mud weight in pounds per gallon (ppg)")
    tvd_ft: float = Field(..., description="True vertical depth in feet")

    model_config = {"json_schema_extra": {"example": {
        "mud_weight_ppg": 10.0, "tvd_ft": 10000.0,
    }}}


class ECDRequest(BaseModel):
    mud_weight_ppg: float = Field(..., description="Static mud weight (ppg)")
    annular_pressure_loss_psi: float = Field(..., description="Annular pressure loss (psi)")
    tvd_ft: float = Field(..., description="True vertical depth (ft)")

    model_config = {"json_schema_extra": {"example": {
        "mud_weight_ppg": 10.0, "annular_pressure_loss_psi": 200.0, "tvd_ft": 10000.0,
    }}}


class KillSheetRequest(BaseModel):
    sidp_psi: float = Field(..., description="Shut-in drill pipe pressure (psi)")
    original_mud_weight_ppg: float = Field(..., description="Original mud weight (ppg)")
    tvd_ft: float = Field(..., description="True vertical depth (ft)")
    circulating_pressure_psi: float = Field(..., description="Slow circulating pressure at kill rate (psi)")

    model_config = {"json_schema_extra": {"example": {
        "sidp_psi": 500.0, "original_mud_weight_ppg": 10.5,
        "tvd_ft": 12000.0, "circulating_pressure_psi": 800.0,
    }}}


@app.post(
    "/api/v1/drilling/hydrostatic",
    tags=["Drilling"],
    summary="Calculate hydrostatic pressure",
    description=(
        "Calculate hydrostatic pressure using: P = 0.052 * mud_weight * TVD. "
        "Fundamental drilling calculation for well control and casing design."
    ),
    response_description="Hydrostatic pressure in psi.",
)
def drilling_hydrostatic(req: HydrostaticRequest):
    """Calculate hydrostatic pressure."""
    from petro_mcp.tools.drilling import calculate_hydrostatic_pressure
    return _call(
        calculate_hydrostatic_pressure,
        mud_weight_ppg=req.mud_weight_ppg, tvd_ft=req.tvd_ft,
    )


@app.post(
    "/api/v1/drilling/ecd",
    tags=["Drilling"],
    summary="Calculate equivalent circulating density",
    description=(
        "Calculate ECD (Equivalent Circulating Density) by adding annular "
        "pressure losses to static mud weight. Critical for avoiding formation "
        "fracture while circulating."
    ),
    response_description="ECD in ppg.",
)
def drilling_ecd(req: ECDRequest):
    """Calculate equivalent circulating density."""
    from petro_mcp.tools.drilling import calculate_ecd
    return _call(
        calculate_ecd,
        mud_weight_ppg=req.mud_weight_ppg,
        annular_pressure_loss_psi=req.annular_pressure_loss_psi,
        tvd_ft=req.tvd_ft,
    )


@app.post(
    "/api/v1/drilling/kill-sheet",
    tags=["Drilling"],
    summary="Calculate kill sheet (well control)",
    description=(
        "Calculate kill mud weight, Initial Circulating Pressure (ICP), and "
        "Final Circulating Pressure (FCP) for the Driller's Method. Essential "
        "well control calculation."
    ),
    response_description="Kill mud weight (ppg), ICP (psi), and FCP (psi).",
)
def drilling_kill_sheet(req: KillSheetRequest):
    """Calculate kill mud weight and ICP/FCP."""
    from petro_mcp.tools.drilling import calculate_kill_mud_weight, calculate_icp_fcp

    kill_result = _call(
        calculate_kill_mud_weight,
        sidp_psi=req.sidp_psi,
        original_mud_weight_ppg=req.original_mud_weight_ppg,
        tvd_ft=req.tvd_ft,
    )

    kill_mw = kill_result["kill_mud_weight_ppg"]

    icp_fcp_result = _call(
        calculate_icp_fcp,
        sidp_psi=req.sidp_psi,
        circulating_pressure_psi=req.circulating_pressure_psi,
        kill_mud_weight_ppg=kill_mw,
        original_mud_weight_ppg=req.original_mud_weight_ppg,
    )

    return {
        **kill_result,
        "icp_psi": icp_fcp_result["icp_psi"],
        "fcp_psi": icp_fcp_result["fcp_psi"],
    }


# ===================================================================
# Economics endpoints
# ===================================================================

class NPVRequest(BaseModel):
    cash_flows: list[float] = Field(
        ...,
        description="Monthly cash flows. First element is typically negative (capex). Subsequent are net revenue.",
    )
    discount_rate: float = Field(0.10, description="Annual discount rate (e.g. 0.10 for 10%)")

    model_config = {"json_schema_extra": {"example": {
        "cash_flows": [-1000000, 50000, 50000, 50000, 50000, 50000,
                       50000, 50000, 50000, 50000, 50000, 50000, 50000],
        "discount_rate": 0.10,
    }}}


class WellEconomicsRequest(BaseModel):
    monthly_oil_bbl: list[float] = Field(..., description="Monthly oil production (bbl)")
    monthly_gas_mcf: list[float] = Field(..., description="Monthly gas production (Mcf)")
    monthly_water_bbl: list[float] = Field(..., description="Monthly water production (bbl)")
    oil_price_bbl: float = Field(..., description="Oil price ($/bbl)")
    gas_price_mcf: float = Field(..., description="Gas price ($/Mcf)")
    opex_monthly: float = Field(..., description="Monthly operating expense ($)")
    capex: float = Field(..., description="Total capital expenditure ($)")
    royalty_pct: float = Field(0.125, description="Royalty fraction (0-1), default 12.5%")
    tax_rate: float = Field(0.0, description="Severance/production tax rate (0-1)")
    discount_rate: float = Field(0.10, description="Annual discount rate (0-1)")
    working_interest: float = Field(1.0, description="Working interest fraction (0-1)")
    net_revenue_interest: float = Field(0.875, description="Net revenue interest fraction (0-1)")

    model_config = {"json_schema_extra": {"example": {
        "monthly_oil_bbl": [500, 475, 451, 428, 407, 387],
        "monthly_gas_mcf": [1000, 950, 902, 857, 814, 773],
        "monthly_water_bbl": [50, 50, 50, 50, 50, 50],
        "oil_price_bbl": 75.0,
        "gas_price_mcf": 3.0,
        "opex_monthly": 5000.0,
        "capex": 500000.0,
    }}}


@app.post(
    "/api/v1/economics/npv",
    tags=["Economics"],
    summary="Calculate Net Present Value",
    description=(
        "Calculate NPV from a series of monthly cash flows at a given annual "
        "discount rate. Converts annual rate to monthly for discounting."
    ),
    response_description="NPV value and discounted cash flow details.",
)
def economics_npv(req: NPVRequest):
    """Calculate Net Present Value from monthly cash flows."""
    from petro_mcp.tools.economics import calculate_npv
    return _call(calculate_npv, cash_flows=req.cash_flows, discount_rate=req.discount_rate)


@app.post(
    "/api/v1/economics/well-economics",
    tags=["Economics"],
    summary="Full well economics (DCF analysis)",
    description=(
        "Perform a complete discounted cash flow analysis for a well. "
        "Calculates NPV, IRR, payout period, and monthly cash flows accounting "
        "for royalties, severance taxes, working interest, and net revenue interest."
    ),
    response_description="NPV, IRR, payout month, and monthly cash flow breakdown.",
)
def economics_well(req: WellEconomicsRequest):
    """Full discounted cash flow analysis for a well."""
    from petro_mcp.tools.economics import calculate_well_economics
    return _call(
        calculate_well_economics,
        monthly_oil_bbl=req.monthly_oil_bbl,
        monthly_gas_mcf=req.monthly_gas_mcf,
        monthly_water_bbl=req.monthly_water_bbl,
        oil_price_bbl=req.oil_price_bbl,
        gas_price_mcf=req.gas_price_mcf,
        opex_monthly=req.opex_monthly,
        capex=req.capex,
        royalty_pct=req.royalty_pct,
        tax_rate=req.tax_rate,
        discount_rate=req.discount_rate,
        working_interest=req.working_interest,
        net_revenue_interest=req.net_revenue_interest,
    )


# ===================================================================
# System / catalog endpoints
# ===================================================================

@app.get("/health", tags=["System"])
def health():
    """Health check for load balancers and uptime monitors."""
    return {"status": "ok", "service": "petro-mcp-api", "version": "0.7.0"}


@app.get(
    "/api/v1/docs",
    tags=["System"],
    summary="API endpoint catalog",
    description=(
        "Returns a JSON catalog of every available endpoint with method, path, "
        "summary, description, and tag. Useful for programmatic discovery."
    ),
    response_description="List of all API endpoints.",
)
def api_catalog():
    """Return the full endpoint catalog as JSON for programmatic discovery."""
    catalog: list[dict[str, Any]] = []
    for route in app.routes:
        # Only include APIRoutes (skip static mounts, etc.)
        if not hasattr(route, "methods"):
            continue
        methods = sorted(route.methods - {"HEAD", "OPTIONS"})  # type: ignore[union-attr]
        if not methods:
            continue
        endpoint_func = getattr(route, "endpoint", None)
        catalog.append({
            "path": route.path,  # type: ignore[union-attr]
            "methods": methods,
            "summary": getattr(route, "summary", None) or "",
            "description": getattr(route, "description", None) or (
                endpoint_func.__doc__ if endpoint_func else ""
            ),
            "tags": getattr(route, "tags", []) or [],
        })
    return {
        "total_endpoints": len(catalog),
        "version": "0.7.0",
        "endpoints": catalog,
    }


# ===================================================================
# Entry point
# ===================================================================

def main():
    """Run the API server (entry point for `petro-api` command)."""
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "petro_mcp.api.app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
