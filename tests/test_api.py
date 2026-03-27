"""Tests for the petro-mcp REST API.

Uses FastAPI TestClient to hit every endpoint group without starting
a real server.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from petro_mcp.api.app import app

client = TestClient(app)


# ===================================================================
# Health
# ===================================================================

class TestHealth:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "version" in body


# ===================================================================
# DCA
# ===================================================================

class TestDecline:
    PRODUCTION_DATA = [
        {"time": 0, "rate": 1000},
        {"time": 1, "rate": 900},
        {"time": 2, "rate": 820},
        {"time": 3, "rate": 750},
        {"time": 4, "rate": 690},
        {"time": 5, "rate": 640},
        {"time": 6, "rate": 600},
        {"time": 7, "rate": 560},
        {"time": 8, "rate": 525},
        {"time": 9, "rate": 495},
        {"time": 10, "rate": 470},
        {"time": 11, "rate": 445},
    ]

    def test_fit_decline_hyperbolic(self):
        r = client.post("/api/v1/decline/fit", json={
            "production_data": self.PRODUCTION_DATA,
            "model": "hyperbolic",
        })
        assert r.status_code == 200
        body = r.json()
        assert "parameters" in body
        assert body["model"] == "hyperbolic"
        assert body["r_squared"] > 0.9

    def test_fit_decline_exponential(self):
        r = client.post("/api/v1/decline/fit", json={
            "production_data": self.PRODUCTION_DATA,
            "model": "exponential",
        })
        assert r.status_code == 200
        assert r.json()["model"] == "exponential"

    def test_fit_decline_bad_model(self):
        r = client.post("/api/v1/decline/fit", json={
            "production_data": self.PRODUCTION_DATA,
            "model": "invalid_model",
        })
        assert r.status_code == 400

    def test_eur(self):
        r = client.post("/api/v1/decline/eur", json={
            "qi": 800, "Di": 0.06, "b": 1.2, "model": "hyperbolic",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["eur"] > 0
        assert body["model"] == "hyperbolic"

    def test_eur_bad_qi(self):
        r = client.post("/api/v1/decline/eur", json={
            "qi": -100, "Di": 0.06, "b": 1.2,
        })
        assert r.status_code == 400

    def test_forecast(self):
        r = client.post("/api/v1/decline/forecast", json={
            "qi": 500, "Di": 0.05, "b": 1.0,
            "model": "hyperbolic", "months": 60,
        })
        assert r.status_code == 200
        body = r.json()
        assert len(body["rates"]) == 61  # 0..60
        assert body["rates"][0] > body["rates"][-1]


# ===================================================================
# PVT
# ===================================================================

class TestPVT:
    def test_pvt_properties_standing(self):
        r = client.post("/api/v1/pvt/properties", json={
            "api_gravity": 35.0,
            "gas_sg": 0.75,
            "temperature": 200.0,
            "pressure": 3000.0,
        })
        assert r.status_code == 200
        body = r.json()
        assert "oil_properties" in body or "bubble_point_pressure_psi" in body

    def test_bubble_point(self):
        r = client.post("/api/v1/pvt/bubble-point", json={
            "api_gravity": 35.0,
            "gas_sg": 0.75,
            "temperature": 200.0,
            "rs": 500.0,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["bubble_point_pressure_psi"] > 0

    def test_z_factor(self):
        r = client.post("/api/v1/pvt/z-factor", json={
            "temperature": 200.0,
            "pressure": 2000.0,
            "gas_sg": 0.70,
        })
        assert r.status_code == 200
        body = r.json()
        assert 0 < body["z_factor"] < 2

    def test_pvt_bad_gravity(self):
        r = client.post("/api/v1/pvt/properties", json={
            "api_gravity": -5.0,
            "gas_sg": 0.75,
            "temperature": 200.0,
            "pressure": 3000.0,
        })
        assert r.status_code == 400


# ===================================================================
# Petrophysics
# ===================================================================

class TestPetrophysics:
    def test_archie(self):
        r = client.post("/api/v1/petrophys/archie", json={
            "rt": 20.0, "phi": 0.20, "rw": 0.05,
        })
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["water_saturation"] <= 1

    def test_density_porosity(self):
        r = client.post("/api/v1/petrophys/porosity", json={
            "rhob": 2.40,
        })
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["density_porosity"] <= 1

    def test_vshale_linear(self):
        r = client.post("/api/v1/petrophys/vshale", json={
            "gr": 70.0, "gr_clean": 20.0, "gr_shale": 120.0,
        })
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["vshale"] <= 1

    def test_vshale_bad_method(self):
        r = client.post("/api/v1/petrophys/vshale", json={
            "gr": 70.0, "gr_clean": 20.0, "gr_shale": 120.0,
            "method": "bogus",
        })
        assert r.status_code == 400


# ===================================================================
# Drilling
# ===================================================================

class TestDrilling:
    def test_hydrostatic(self):
        r = client.post("/api/v1/drilling/hydrostatic", json={
            "mud_weight_ppg": 10.0, "tvd_ft": 10000.0,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["hydrostatic_pressure_psi"] == pytest.approx(5200.0, rel=0.01)

    def test_ecd(self):
        r = client.post("/api/v1/drilling/ecd", json={
            "mud_weight_ppg": 10.0,
            "annular_pressure_loss_psi": 200.0,
            "tvd_ft": 10000.0,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["ecd_ppg"] > 10.0

    def test_kill_sheet(self):
        r = client.post("/api/v1/drilling/kill-sheet", json={
            "sidp_psi": 500.0,
            "original_mud_weight_ppg": 10.5,
            "tvd_ft": 12000.0,
            "circulating_pressure_psi": 800.0,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["kill_mud_weight_ppg"] > 10.5
        assert "icp_psi" in body
        assert "fcp_psi" in body


# ===================================================================
# Economics
# ===================================================================

class TestEconomics:
    def test_npv(self):
        r = client.post("/api/v1/economics/npv", json={
            "cash_flows": [-1000000, 50000, 50000, 50000, 50000, 50000,
                           50000, 50000, 50000, 50000, 50000, 50000, 50000],
            "discount_rate": 0.10,
        })
        assert r.status_code == 200
        body = r.json()
        assert "npv" in body

    def test_well_economics(self):
        n = 36
        oil = [500 * (0.95 ** i) for i in range(n)]
        gas = [g * 2.0 for g in oil]
        water = [50.0] * n
        r = client.post("/api/v1/economics/well-economics", json={
            "monthly_oil_bbl": oil,
            "monthly_gas_mcf": gas,
            "monthly_water_bbl": water,
            "oil_price_bbl": 75.0,
            "gas_price_mcf": 3.0,
            "opex_monthly": 5000.0,
            "capex": 500000.0,
        })
        assert r.status_code == 200
        body = r.json()
        assert "npv" in body
        assert "irr_pct" in body

    def test_npv_empty(self):
        r = client.post("/api/v1/economics/npv", json={
            "cash_flows": [],
        })
        assert r.status_code == 400
