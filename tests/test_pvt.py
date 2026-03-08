"""Tests for PVT (Pressure-Volume-Temperature) correlation tools."""

import json
import math

import pytest

from petro_mcp.tools.pvt import bubble_point, calculate_pvt


class TestCalculatePVT:
    """Tests for the comprehensive PVT calculation function."""

    def test_typical_black_oil(self):
        """Test with typical black oil: 35 API, 0.65 gas SG, 200°F, 3000 psi."""
        result = json.loads(calculate_pvt(
            api_gravity=35,
            gas_sg=0.65,
            temperature=200,
            pressure=3000,
        ))

        oil = result["oil_properties"]
        gas = result["gas_properties"]

        # Bubble point should be in reasonable range for typical black oil
        assert 500 <= oil["bubble_point_pressure_psi"] <= 5000

        # Rs should be positive at 3000 psi
        assert oil["solution_gor_scf_stb"] > 0

        # Bo must always be > 1.0 for live oil (gas in solution expands oil)
        assert oil["oil_fvf_bbl_stb"] > 1.0

        # Oil density at reservoir conditions (typically 30-55 lb/ft³)
        assert 25 <= oil["oil_density_lb_ft3"] <= 60

        # Viscosities must be positive
        assert oil["dead_oil_viscosity_cp"] > 0
        assert oil["live_oil_viscosity_cp"] > 0

        # Live oil viscosity < dead oil viscosity (gas reduces viscosity)
        assert oil["live_oil_viscosity_cp"] < oil["dead_oil_viscosity_cp"]

        # Z-factor in physical range
        assert 0.2 <= gas["z_factor"] <= 1.5

        # Gas FVF must be positive
        assert gas["gas_fvf_rcf_scf"] > 0

        # Gas viscosity must be positive (typically 0.01 - 0.05 cp)
        assert gas["gas_viscosity_cp"] > 0

        # Gas compressibility must be positive
        assert gas["gas_compressibility_1_psi"] > 0

    def test_light_oil_high_pressure(self):
        """Test with light oil at high pressure."""
        result = json.loads(calculate_pvt(
            api_gravity=45,
            gas_sg=0.70,
            temperature=250,
            pressure=5000,
        ))
        oil = result["oil_properties"]
        gas = result["gas_properties"]

        assert oil["oil_fvf_bbl_stb"] > 1.0
        assert oil["solution_gor_scf_stb"] > 0
        assert 0.2 <= gas["z_factor"] <= 1.5

    def test_heavy_oil(self):
        """Test with heavy oil (low API)."""
        result = json.loads(calculate_pvt(
            api_gravity=15,
            gas_sg=0.65,
            temperature=150,
            pressure=2000,
        ))
        oil = result["oil_properties"]

        # Heavy oil should have higher viscosity than light oil
        assert oil["dead_oil_viscosity_cp"] > 1.0
        assert oil["oil_fvf_bbl_stb"] > 1.0

    def test_rs_near_atmospheric(self):
        """At atmospheric pressure, Rs should be very small (near zero)."""
        result = json.loads(calculate_pvt(
            api_gravity=35,
            gas_sg=0.65,
            temperature=200,
            pressure=14.7,
        ))
        oil = result["oil_properties"]

        # Rs at atmospheric should be negligible
        assert oil["solution_gor_scf_stb"] < 5.0

    def test_correlations_identified(self):
        """Verify all correlations are cited in the output."""
        result = json.loads(calculate_pvt(
            api_gravity=35,
            gas_sg=0.65,
            temperature=200,
            pressure=3000,
        ))
        oil = result["oil_properties"]
        gas = result["gas_properties"]

        assert "Standing" in oil["bubble_point_correlation"]
        assert "Standing" in oil["solution_gor_correlation"]
        assert "Standing" in oil["oil_fvf_correlation"]
        assert "Beggs" in oil["oil_viscosity_correlation"]
        assert "Hall" in gas["z_factor_correlation"]
        assert "Lee" in gas["gas_viscosity_correlation"]
        assert "Sutton" in gas["pseudocritical_correlation"]

    def test_units_present(self):
        """Verify units are included in the output."""
        result = json.loads(calculate_pvt(
            api_gravity=35,
            gas_sg=0.65,
            temperature=200,
            pressure=3000,
        ))
        units = result["units"]

        assert units["bubble_point_pressure"] == "psi"
        assert units["solution_gor"] == "scf/STB"
        assert units["oil_fvf"] == "bbl/STB"
        assert units["z_factor"] == "dimensionless"
        assert units["gas_fvf"] == "rcf/scf"
        assert units["gas_viscosity"] == "cp"
        assert units["gas_compressibility"] == "1/psi"

    def test_inputs_echoed(self):
        """Verify inputs are echoed back in the result."""
        result = json.loads(calculate_pvt(
            api_gravity=35,
            gas_sg=0.65,
            temperature=200,
            pressure=3000,
            separator_pressure=150,
            separator_temperature=75,
        ))
        inputs = result["inputs"]
        assert inputs["api_gravity"] == 35
        assert inputs["gas_sg"] == 0.65
        assert inputs["temperature_F"] == 200
        assert inputs["pressure_psi"] == 3000
        assert inputs["separator_pressure_psi"] == 150
        assert inputs["separator_temperature_F"] == 75

    def test_gas_density_positive(self):
        """Gas density must be positive at reservoir conditions."""
        result = json.loads(calculate_pvt(
            api_gravity=35,
            gas_sg=0.65,
            temperature=200,
            pressure=3000,
        ))
        assert result["gas_properties"]["gas_density_lb_ft3"] > 0

    def test_default_separator_conditions(self):
        """Verify default separator conditions are used when not specified."""
        result = json.loads(calculate_pvt(
            api_gravity=35,
            gas_sg=0.65,
            temperature=200,
            pressure=3000,
        ))
        assert result["inputs"]["separator_pressure_psi"] == 100.0
        assert result["inputs"]["separator_temperature_F"] == 60.0


class TestCalculatePVTInvalidInputs:
    """Tests for invalid input handling."""

    def test_negative_api_gravity(self):
        with pytest.raises(ValueError, match="API gravity must be positive"):
            calculate_pvt(api_gravity=-10, gas_sg=0.65, temperature=200, pressure=3000)

    def test_zero_api_gravity(self):
        with pytest.raises(ValueError, match="API gravity must be positive"):
            calculate_pvt(api_gravity=0, gas_sg=0.65, temperature=200, pressure=3000)

    def test_negative_gas_sg(self):
        with pytest.raises(ValueError, match="Gas specific gravity must be positive"):
            calculate_pvt(api_gravity=35, gas_sg=-0.5, temperature=200, pressure=3000)

    def test_negative_temperature(self):
        with pytest.raises(ValueError, match="Temperature must be positive"):
            calculate_pvt(api_gravity=35, gas_sg=0.65, temperature=-100, pressure=3000)

    def test_negative_pressure(self):
        with pytest.raises(ValueError, match="Pressure must be positive"):
            calculate_pvt(api_gravity=35, gas_sg=0.65, temperature=200, pressure=-500)

    def test_zero_pressure(self):
        with pytest.raises(ValueError, match="Pressure must be positive"):
            calculate_pvt(api_gravity=35, gas_sg=0.65, temperature=200, pressure=0)


class TestBubblePoint:
    """Tests for the standalone bubble point function."""

    def test_typical_values(self):
        """Bubble point for typical oil with known Rs."""
        result = json.loads(bubble_point(
            api_gravity=35,
            gas_sg=0.65,
            temperature=200,
            rs=500,
        ))
        pb = result["bubble_point_pressure_psi"]

        # For 35 API oil, 0.65 gas SG, 200°F, Rs=500 scf/STB
        # Pb should be in the range of ~1500-3500 psi
        assert 1000 <= pb <= 5000

    def test_zero_rs(self):
        """With zero Rs, Pb should be at atmospheric."""
        result = json.loads(bubble_point(
            api_gravity=35,
            gas_sg=0.65,
            temperature=200,
            rs=0,
        ))
        assert result["bubble_point_pressure_psi"] == 14.7

    def test_high_rs(self):
        """Higher Rs should give higher Pb."""
        result_low = json.loads(bubble_point(
            api_gravity=35, gas_sg=0.65, temperature=200, rs=200,
        ))
        result_high = json.loads(bubble_point(
            api_gravity=35, gas_sg=0.65, temperature=200, rs=800,
        ))
        assert result_high["bubble_point_pressure_psi"] > result_low["bubble_point_pressure_psi"]

    def test_correlation_cited(self):
        result = json.loads(bubble_point(
            api_gravity=35, gas_sg=0.65, temperature=200, rs=500,
        ))
        assert "Standing" in result["correlation"]
        assert "1947" in result["correlation"]

    def test_inputs_echoed(self):
        result = json.loads(bubble_point(
            api_gravity=35, gas_sg=0.65, temperature=200, rs=500,
        ))
        assert result["inputs"]["api_gravity"] == 35
        assert result["inputs"]["gas_sg"] == 0.65
        assert result["inputs"]["temperature_F"] == 200
        assert result["inputs"]["solution_gor_scf_stb"] == 500

    def test_negative_api_raises(self):
        with pytest.raises(ValueError, match="API gravity must be positive"):
            bubble_point(api_gravity=-5, gas_sg=0.65, temperature=200, rs=500)

    def test_negative_pressure_raises(self):
        with pytest.raises(ValueError, match="Gas specific gravity must be positive"):
            bubble_point(api_gravity=35, gas_sg=-1, temperature=200, rs=500)

    def test_negative_rs_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            bubble_point(api_gravity=35, gas_sg=0.65, temperature=200, rs=-100)


class TestPVTPhysicalConsistency:
    """Tests that verify physical consistency across conditions."""

    def test_bo_increases_with_pressure_below_pb(self):
        """Bo should increase as pressure increases (below bubble point),
        because more gas dissolves into oil."""
        result_low = json.loads(calculate_pvt(
            api_gravity=35, gas_sg=0.65, temperature=200, pressure=500,
        ))
        result_high = json.loads(calculate_pvt(
            api_gravity=35, gas_sg=0.65, temperature=200, pressure=1500,
        ))
        assert (result_high["oil_properties"]["oil_fvf_bbl_stb"]
                >= result_low["oil_properties"]["oil_fvf_bbl_stb"])

    def test_rs_increases_with_pressure(self):
        """Rs should increase with increasing pressure (below Pb)."""
        result_low = json.loads(calculate_pvt(
            api_gravity=35, gas_sg=0.65, temperature=200, pressure=500,
        ))
        result_high = json.loads(calculate_pvt(
            api_gravity=35, gas_sg=0.65, temperature=200, pressure=2000,
        ))
        assert (result_high["oil_properties"]["solution_gor_scf_stb"]
                > result_low["oil_properties"]["solution_gor_scf_stb"])

    def test_z_factor_range_across_pressures(self):
        """Z-factor should remain in physical range across pressures."""
        for p in [100, 500, 1000, 2000, 4000, 6000]:
            result = json.loads(calculate_pvt(
                api_gravity=35, gas_sg=0.65, temperature=200, pressure=p,
            ))
            z = result["gas_properties"]["z_factor"]
            assert 0.2 <= z <= 1.5, f"Z={z} out of range at P={p} psi"

    def test_gas_fvf_decreases_with_pressure(self):
        """Gas FVF should decrease as pressure increases (gas compresses)."""
        result_low = json.loads(calculate_pvt(
            api_gravity=35, gas_sg=0.65, temperature=200, pressure=500,
        ))
        result_high = json.loads(calculate_pvt(
            api_gravity=35, gas_sg=0.65, temperature=200, pressure=3000,
        ))
        assert (result_low["gas_properties"]["gas_fvf_rcf_scf"]
                > result_high["gas_properties"]["gas_fvf_rcf_scf"])

    def test_heavy_oil_more_viscous(self):
        """Heavy oil (low API) should be more viscous than light oil (high API)."""
        result_heavy = json.loads(calculate_pvt(
            api_gravity=15, gas_sg=0.65, temperature=200, pressure=2000,
        ))
        result_light = json.loads(calculate_pvt(
            api_gravity=45, gas_sg=0.65, temperature=200, pressure=2000,
        ))
        assert (result_heavy["oil_properties"]["dead_oil_viscosity_cp"]
                > result_light["oil_properties"]["dead_oil_viscosity_cp"])
