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
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="petro-mcp API",
    description="REST API for petro-mcp calculation engines. Powers tools.petropt.com.",
    version="0.6.0",
)

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
        ..., description="List of dicts with 'time'+'rate' or 'oil'/'gas' keys"
    )
    model: str = Field("hyperbolic", description="exponential, hyperbolic, harmonic, modified_hyperbolic, duong")


class EURRequest(BaseModel):
    qi: float = Field(..., description="Initial production rate")
    Di: float = Field(0.0, description="Initial decline rate (1/month)")
    b: float = Field(0.0, description="Arps b-factor")
    economic_limit: float = Field(5.0, description="Minimum economic rate")
    model: str = Field("hyperbolic", description="Decline model name")
    max_time: float = Field(600, description="Max time in months")
    Dmin: float = Field(0.005, description="Terminal decline rate for modified_hyperbolic")
    a: float = Field(1.0, description="Duong intercept parameter")
    m: float = Field(1.1, description="Duong slope parameter")


class ForecastRequest(BaseModel):
    qi: float = Field(..., description="Initial production rate")
    Di: float = Field(0.0, description="Initial decline rate (1/month)")
    b: float = Field(0.0, description="Arps b-factor")
    model: str = Field("hyperbolic", description="Decline model name")
    months: int = Field(360, description="Number of months to forecast")
    Dmin: float = Field(0.005, description="Terminal decline for modified_hyperbolic")
    a: float = Field(1.0, description="Duong intercept parameter")
    m: float = Field(1.1, description="Duong slope parameter")


@app.post("/api/v1/decline/fit", tags=["DCA"])
def fit_decline(req: FitDeclineRequest):
    """Fit an Arps decline curve to production data."""
    from petro_mcp.tools.decline import fit_decline_curve
    return _call(fit_decline_curve, production_data=req.production_data, model=req.model)


@app.post("/api/v1/decline/eur", tags=["DCA"])
def decline_eur(req: EURRequest):
    """Calculate Estimated Ultimate Recovery from decline parameters."""
    from petro_mcp.tools.decline import calculate_eur
    return _call(
        calculate_eur,
        qi=req.qi, Di=req.Di, b=req.b,
        economic_limit=req.economic_limit, model=req.model,
        max_time=req.max_time, Dmin=req.Dmin, a=req.a, m=req.m,
    )


@app.post("/api/v1/decline/forecast", tags=["DCA"])
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
    api_gravity: float = Field(..., description="Oil API gravity")
    gas_sg: float = Field(..., description="Gas specific gravity (air=1.0)")
    temperature: float = Field(..., description="Reservoir temperature (F)")
    pressure: float = Field(..., description="Current pressure (psi)")
    separator_pressure: float = Field(100.0, description="Separator pressure (psi)")
    separator_temperature: float = Field(60.0, description="Separator temperature (F)")
    correlation: str = Field("standing", description="standing, vasquez_beggs, or petrosky_farshad")


class BubblePointRequest(BaseModel):
    api_gravity: float = Field(..., description="Oil API gravity")
    gas_sg: float = Field(..., description="Gas specific gravity")
    temperature: float = Field(..., description="Temperature (F)")
    rs: float = Field(..., description="Solution GOR (scf/STB)")


class GasZRequest(BaseModel):
    temperature: float = Field(..., description="Temperature (F)")
    pressure: float = Field(..., description="Pressure (psi)")
    gas_sg: float = Field(..., description="Gas specific gravity")
    method: str = Field("hall_yarborough", description="hall_yarborough or dranchuk_abou_kassem")
    pseudocritical_method: str = Field("sutton", description="sutton or piper")
    h2s_fraction: float = Field(0.0, description="Mole fraction H2S")
    co2_fraction: float = Field(0.0, description="Mole fraction CO2")
    n2_fraction: float = Field(0.0, description="Mole fraction N2")


@app.post("/api/v1/pvt/properties", tags=["PVT"])
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


@app.post("/api/v1/pvt/bubble-point", tags=["PVT"])
def pvt_bubble_point(req: BubblePointRequest):
    """Calculate bubble point pressure (Standing)."""
    from petro_mcp.tools.pvt import bubble_point
    return _call(
        bubble_point,
        api_gravity=req.api_gravity, gas_sg=req.gas_sg,
        temperature=req.temperature, rs=req.rs,
    )


@app.post("/api/v1/pvt/z-factor", tags=["PVT"])
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
    rt: float = Field(..., description="True resistivity (ohm-m)")
    phi: float = Field(..., description="Porosity (fraction 0-1)")
    rw: float = Field(..., description="Formation water resistivity (ohm-m)")
    a: float = Field(1.0, description="Tortuosity factor")
    m: float = Field(2.0, description="Cementation exponent")
    n: float = Field(2.0, description="Saturation exponent")


class DensityPorosityRequest(BaseModel):
    rhob: float = Field(..., description="Bulk density (g/cc)")
    rho_matrix: float = Field(2.65, description="Matrix density (g/cc)")
    rho_fluid: float = Field(1.0, description="Fluid density (g/cc)")


class VshaleRequest(BaseModel):
    gr: float = Field(..., description="Gamma ray reading (API)")
    gr_clean: float = Field(..., description="GR in clean sand (API)")
    gr_shale: float = Field(..., description="GR in pure shale (API)")
    method: str = Field("linear", description="linear, larionov_tertiary, larionov_older, clavier")


@app.post("/api/v1/petrophys/archie", tags=["Petrophysics"])
def petrophys_archie(req: ArchieRequest):
    """Calculate water saturation using Archie equation."""
    from petro_mcp.tools.petrophysics import calculate_archie_sw
    return _call(
        calculate_archie_sw,
        rt=req.rt, phi=req.phi, rw=req.rw,
        a=req.a, m=req.m, n=req.n,
    )


@app.post("/api/v1/petrophys/porosity", tags=["Petrophysics"])
def petrophys_porosity(req: DensityPorosityRequest):
    """Calculate porosity from bulk density."""
    from petro_mcp.tools.petrophysics import calculate_density_porosity
    return _call(
        calculate_density_porosity,
        rhob=req.rhob, rho_matrix=req.rho_matrix, rho_fluid=req.rho_fluid,
    )


@app.post("/api/v1/petrophys/vshale", tags=["Petrophysics"])
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
    mud_weight_ppg: float = Field(..., description="Mud weight (ppg)")
    tvd_ft: float = Field(..., description="True vertical depth (ft)")


class ECDRequest(BaseModel):
    mud_weight_ppg: float = Field(..., description="Static mud weight (ppg)")
    annular_pressure_loss_psi: float = Field(..., description="Annular pressure loss (psi)")
    tvd_ft: float = Field(..., description="True vertical depth (ft)")


class KillSheetRequest(BaseModel):
    sidp_psi: float = Field(..., description="Shut-in drill pipe pressure (psi)")
    original_mud_weight_ppg: float = Field(..., description="Original mud weight (ppg)")
    tvd_ft: float = Field(..., description="True vertical depth (ft)")
    circulating_pressure_psi: float = Field(..., description="Slow circulating pressure (psi)")


@app.post("/api/v1/drilling/hydrostatic", tags=["Drilling"])
def drilling_hydrostatic(req: HydrostaticRequest):
    """Calculate hydrostatic pressure."""
    from petro_mcp.tools.drilling import calculate_hydrostatic_pressure
    return _call(
        calculate_hydrostatic_pressure,
        mud_weight_ppg=req.mud_weight_ppg, tvd_ft=req.tvd_ft,
    )


@app.post("/api/v1/drilling/ecd", tags=["Drilling"])
def drilling_ecd(req: ECDRequest):
    """Calculate equivalent circulating density."""
    from petro_mcp.tools.drilling import calculate_ecd
    return _call(
        calculate_ecd,
        mud_weight_ppg=req.mud_weight_ppg,
        annular_pressure_loss_psi=req.annular_pressure_loss_psi,
        tvd_ft=req.tvd_ft,
    )


@app.post("/api/v1/drilling/kill-sheet", tags=["Drilling"])
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
    cash_flows: list[float] = Field(..., description="Monthly cash flows (first is typically negative capex)")
    discount_rate: float = Field(0.10, description="Annual discount rate")


class WellEconomicsRequest(BaseModel):
    monthly_oil_bbl: list[float] = Field(..., description="Monthly oil production (bbl)")
    monthly_gas_mcf: list[float] = Field(..., description="Monthly gas production (Mcf)")
    monthly_water_bbl: list[float] = Field(..., description="Monthly water production (bbl)")
    oil_price_bbl: float = Field(..., description="Oil price ($/bbl)")
    gas_price_mcf: float = Field(..., description="Gas price ($/Mcf)")
    opex_monthly: float = Field(..., description="Monthly operating expense ($)")
    capex: float = Field(..., description="Total capital expenditure ($)")
    royalty_pct: float = Field(0.125, description="Royalty fraction (0-1)")
    tax_rate: float = Field(0.0, description="Severance/production tax rate (0-1)")
    discount_rate: float = Field(0.10, description="Annual discount rate")
    working_interest: float = Field(1.0, description="Working interest fraction (0-1)")
    net_revenue_interest: float = Field(0.875, description="Net revenue interest fraction (0-1)")


@app.post("/api/v1/economics/npv", tags=["Economics"])
def economics_npv(req: NPVRequest):
    """Calculate Net Present Value from monthly cash flows."""
    from petro_mcp.tools.economics import calculate_npv
    return _call(calculate_npv, cash_flows=req.cash_flows, discount_rate=req.discount_rate)


@app.post("/api/v1/economics/well-economics", tags=["Economics"])
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
# Health check
# ===================================================================

@app.get("/health", tags=["System"])
def health():
    """Health check for load balancers and uptime monitors."""
    return {"status": "ok", "service": "petro-mcp-api", "version": "0.6.0"}


# ===================================================================
# Entry point
# ===================================================================

def main():
    """Run the API server (entry point for `petro-api` command)."""
    uvicorn.run(
        "petro_mcp.api.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
