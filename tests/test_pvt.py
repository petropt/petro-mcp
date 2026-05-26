"""Tests for PVT (Pressure-Volume-Temperature) correlation tools."""

import json
import math

import pytest

from petro_mcp.tools.pvt import (
    bubble_point,
    calculate_gas_z_factor,
    calculate_pvt,
)


# ===========================================================================
# calculate_pvt — Standing (default)
# ===========================================================================

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
        ))
        inputs = result["inputs"]
        assert inputs["api_gravity"] == 35
        assert inputs["gas_sg"] == 0.65
        assert inputs["temperature_F"] == 200
        assert inputs["pressure_psi"] == 3000
        assert inputs["separator_pressure_psi"] == 150

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
        """Verify default separator pressure is used when not specified."""
        result = json.loads(calculate_pvt(
            api_gravity=35,
            gas_sg=0.65,
            temperature=200,
            pressure=3000,
        ))
        assert result["inputs"]["separator_pressure_psi"] == 100.0

    def test_default_correlation_is_standing(self):
        """Default correlation should be standing."""
        result = json.loads(calculate_pvt(
            api_gravity=35, gas_sg=0.65, temperature=200, pressure=3000,
        ))
        assert result["inputs"]["correlation"] == "standing"


# ===========================================================================
# calculate_pvt — Vasquez-Beggs correlation
# ===========================================================================

class TestCalculatePVTVasquezBeggs:
    """Tests for calculate_pvt with Vasquez-Beggs oil correlations."""

    def test_typical_values(self):
        result = json.loads(calculate_pvt(
            api_gravity=35, gas_sg=0.65, temperature=200, pressure=3000,
            correlation="vasquez_beggs",
        ))
        oil = result["oil_properties"]
        assert oil["solution_gor_scf_stb"] > 0
        assert oil["oil_fvf_bbl_stb"] > 1.0
        # VB uses Standing's Pb correlation internally
        assert "Standing" in oil["bubble_point_correlation"]

    def test_heavy_oil_coefficients(self):
        """API <= 30 uses different coefficients."""
        result = json.loads(calculate_pvt(
            api_gravity=25, gas_sg=0.70, temperature=180, pressure=2500,
            correlation="vasquez_beggs",
        ))
        oil = result["oil_properties"]
        assert oil["solution_gor_scf_stb"] > 0
        assert oil["oil_fvf_bbl_stb"] >= 1.0

    def test_light_oil_coefficients(self):
        """API > 30 uses different coefficients."""
        result = json.loads(calculate_pvt(
            api_gravity=40, gas_sg=0.70, temperature=220, pressure=4000,
            correlation="vasquez_beggs",
        ))
        oil = result["oil_properties"]
        assert oil["solution_gor_scf_stb"] > 0
        assert oil["oil_fvf_bbl_stb"] > 1.0

    def test_correlation_echoed(self):
        result = json.loads(calculate_pvt(
            api_gravity=35, gas_sg=0.65, temperature=200, pressure=3000,
            correlation="vasquez_beggs",
        ))
        assert result["inputs"]["correlation"] == "vasquez_beggs"


# ===========================================================================
# calculate_pvt — Petrosky-Farshad correlation
# ===========================================================================

class TestCalculatePVTPetroskyFarshad:
    """Tests for calculate_pvt with Petrosky-Farshad oil correlations."""

    def test_typical_gom_oil(self):
        """Typical Gulf of Mexico oil."""
        result = json.loads(calculate_pvt(
            api_gravity=32, gas_sg=0.70, temperature=210, pressure=3500,
            correlation="petrosky_farshad",
        ))
        oil = result["oil_properties"]
        assert oil["solution_gor_scf_stb"] > 0
        assert oil["oil_fvf_bbl_stb"] > 1.0
        assert "Petrosky" in oil["bubble_point_correlation"]

    def test_light_oil(self):
        result = json.loads(calculate_pvt(
            api_gravity=42, gas_sg=0.65, temperature=180, pressure=2000,
            correlation="petrosky_farshad",
        ))
        oil = result["oil_properties"]
        assert oil["oil_fvf_bbl_stb"] > 1.0

    def test_correlation_echoed(self):
        result = json.loads(calculate_pvt(
            api_gravity=35, gas_sg=0.65, temperature=200, pressure=3000,
            correlation="petrosky_farshad",
        ))
        assert result["inputs"]["correlation"] == "petrosky_farshad"
        assert "Petrosky" in result["oil_properties"]["oil_fvf_correlation"]


# ===========================================================================
# calculate_pvt — invalid correlation
# ===========================================================================

class TestCalculatePVTInvalidCorrelation:
    def test_unknown_correlation_raises(self):
        with pytest.raises(ValueError, match="Unknown correlation"):
            calculate_pvt(
                api_gravity=35, gas_sg=0.65, temperature=200, pressure=3000,
                correlation="invalid_name",
            )


# ===========================================================================
# calculate_pvt — invalid inputs
# ===========================================================================

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


# ===========================================================================
# bubble_point (standalone)
# ===========================================================================

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


# ===========================================================================
# Gas Z-factor (standalone with method choice)
# ===========================================================================

class TestGasZFactor:
    """Tests for calculate_gas_z_factor."""

    def test_hall_yarborough_default(self):
        result = json.loads(calculate_gas_z_factor(
            temperature=200, pressure=3000, gas_sg=0.65,
        ))
        z = result["z_factor"]
        assert 0.2 <= z <= 1.5
        assert "Hall" in result["z_factor_correlation"]

    def test_dranchuk_abou_kassem(self):
        result = json.loads(calculate_gas_z_factor(
            temperature=200, pressure=3000, gas_sg=0.65,
            method="dranchuk_abou_kassem",
        ))
        z = result["z_factor"]
        assert 0.2 <= z <= 1.5
        assert "Dranchuk" in result["z_factor_correlation"]

    def test_dak_and_hy_agree_roughly(self):
        """Both Z-factor methods should give similar results."""
        hy = json.loads(calculate_gas_z_factor(
            temperature=200, pressure=2000, gas_sg=0.70,
            method="hall_yarborough",
        ))
        dak = json.loads(calculate_gas_z_factor(
            temperature=200, pressure=2000, gas_sg=0.70,
            method="dranchuk_abou_kassem",
        ))
        # Should be within 10% of each other
        z_hy = hy["z_factor"]
        z_dak = dak["z_factor"]
        assert abs(z_hy - z_dak) / z_hy < 0.10

    def test_piper_pseudocritical(self):
        result = json.loads(calculate_gas_z_factor(
            temperature=200, pressure=3000, gas_sg=0.65,
            pseudocritical_method="piper",
        ))
        assert "Piper" in result["pseudocritical_correlation"]
        assert 0.2 <= result["z_factor"] <= 1.5

    def test_sour_gas_with_piper(self):
        """Piper method with H2S and CO2."""
        result = json.loads(calculate_gas_z_factor(
            temperature=200, pressure=3000, gas_sg=0.75,
            pseudocritical_method="piper",
            h2s_fraction=0.05, co2_fraction=0.10,
        ))
        assert 0.05 <= result["z_factor"] <= 2.0
        assert result["inputs"]["h2s_fraction"] == 0.05
        assert result["inputs"]["co2_fraction"] == 0.10

    def test_gas_compressibility_positive(self):
        result = json.loads(calculate_gas_z_factor(
            temperature=200, pressure=3000, gas_sg=0.65,
        ))
        assert result["gas_compressibility_1_psi"] > 0

    def test_gas_fvf_positive(self):
        result = json.loads(calculate_gas_z_factor(
            temperature=200, pressure=3000, gas_sg=0.65,
        ))
        assert result["gas_fvf_rcf_scf"] > 0

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            calculate_gas_z_factor(
                temperature=200, pressure=3000, gas_sg=0.65,
                method="invalid",
            )

    def test_invalid_pseudocritical_method(self):
        with pytest.raises(ValueError, match="Unknown pseudocritical method"):
            calculate_gas_z_factor(
                temperature=200, pressure=3000, gas_sg=0.65,
                pseudocritical_method="invalid",
            )

    def test_invalid_gas_sg(self):
        with pytest.raises(ValueError, match="Gas specific gravity must be positive"):
            calculate_gas_z_factor(
                temperature=200, pressure=3000, gas_sg=0,
            )

    def test_across_pressures(self):
        """Z-factor should remain physical across pressures."""
        for p in [100, 500, 1000, 3000, 6000, 10000]:
            result = json.loads(calculate_gas_z_factor(
                temperature=200, pressure=p, gas_sg=0.65,
                method="dranchuk_abou_kassem",
            ))
            z = result["z_factor"]
            assert 0.05 <= z <= 2.0, f"Z={z} out of range at P={p}"

    def test_hy_piper_differs_from_sutton(self):
        """Hall-Yarborough with Piper pseudocriticals should give a different
        Z-factor than with Sutton, confirming the method is actually routed."""
        sutton = json.loads(calculate_gas_z_factor(
            temperature=200, pressure=3000, gas_sg=0.70,
            method="hall_yarborough", pseudocritical_method="sutton",
        ))
        piper = json.loads(calculate_gas_z_factor(
            temperature=200, pressure=3000, gas_sg=0.70,
            method="hall_yarborough", pseudocritical_method="piper",
        ))
        assert sutton["z_factor"] != piper["z_factor"]
        assert sutton["pseudocritical_temperature_R"] != piper["pseudocritical_temperature_R"]


# ===========================================================================
# Vasquez-Beggs separator validation
# ===========================================================================

class TestVBSeparatorValidation:
    """Vasquez-Beggs should reject non-positive separator pressure."""

    def test_zero_separator_pressure_raises(self):
        with pytest.raises(ValueError, match="Separator pressure must be positive"):
            calculate_pvt(
                api_gravity=35, gas_sg=0.65, temperature=200, pressure=3000,
                separator_pressure=0, correlation="vasquez_beggs",
            )

    def test_negative_separator_pressure_raises(self):
        with pytest.raises(ValueError, match="Separator pressure must be positive"):
            calculate_pvt(
                api_gravity=35, gas_sg=0.65, temperature=200, pressure=3000,
                separator_pressure=-100, correlation="vasquez_beggs",
            )

# ===========================================================================
# Physical consistency checks
# ===========================================================================

