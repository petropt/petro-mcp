"""Tests for petroleum engineering calculation tools."""

import json

import pytest

from petro_mcp.tools.calculations import calculate_ip_ratio, nodal_analysis


class TestCalculateIPRatio:
    def test_basic_ratios(self):
        result = json.loads(calculate_ip_ratio(500, 1500, 200))
        assert result["oil_rate_bopd"] == 500
        assert result["gor_scf_bbl"] == 3000.0
        assert result["wor"] == 0.4
        assert result["total_liquid_bfpd"] == 700
        assert abs(result["water_cut_pct"] - 28.57) < 0.1

    def test_zero_oil(self):
        result = json.loads(calculate_ip_ratio(0, 5000, 100))
        assert result["gor_scf_bbl"] is None
        assert result["wor"] is None
        assert result["water_cut_pct"] == 100.0

    def test_well_type_black_oil(self):
        result = json.loads(calculate_ip_ratio(500, 750, 200))
        assert result["well_type"] == "black oil"

    def test_well_type_gas(self):
        result = json.loads(calculate_ip_ratio(10, 5000, 50))
        assert result["well_type"] == "dry gas"

    def test_negative_rate_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            calculate_ip_ratio(-100, 500, 200)

    def test_well_type_wet_gas_condensate(self):
        result = json.loads(calculate_ip_ratio(10, 500, 5))
        assert result["well_type"] == "wet gas / condensate"

    def test_well_type_volatile_oil(self):
        result = json.loads(calculate_ip_ratio(100, 500, 50))
        assert result["well_type"] == "volatile oil"


class TestNodalAnalysis:
    def test_basic_nodal(self):
        result = json.loads(nodal_analysis(
            reservoir_pressure=4000,
            PI=5.0,
            tubing_size=2.875,
            wellhead_pressure=200,
        ))
        assert result["aof_bopd"] > 0
        assert "ipr_curve" in result
        assert "vlp_curve" in result
        assert len(result["ipr_curve"]["rates_bopd"]) == 21
        assert len(result["vlp_curve"]["rates_bopd"]) == 21

    def test_operating_point_exists(self):
        result = json.loads(nodal_analysis(
            reservoir_pressure=4000,
            PI=5.0,
            tubing_size=2.875,
            wellhead_pressure=200,
        ))
        assert result["operating_point"] is not None
        assert result["operating_point"]["rate_bopd"] > 0
        assert result["operating_point"]["flowing_pressure_psi"] > 0

    def test_invalid_pressure(self):
        with pytest.raises(ValueError, match="positive"):
            nodal_analysis(
                reservoir_pressure=0,
                PI=5.0,
                tubing_size=2.875,
                wellhead_pressure=200,
            )

    def test_invalid_tubing(self):
        with pytest.raises(ValueError, match="positive"):
            nodal_analysis(
                reservoir_pressure=4000,
                PI=5.0,
                tubing_size=0,
                wellhead_pressure=200,
            )

    def test_no_intersection(self):
        """High wellhead pressure prevents natural flow."""
        result = json.loads(nodal_analysis(
            reservoir_pressure=1000,
            PI=0.5,
            tubing_size=2.875,
            wellhead_pressure=3000,
        ))
        assert result["operating_point"] is None
        assert "note" in result
