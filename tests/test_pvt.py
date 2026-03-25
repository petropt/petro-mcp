"""Tests for PVT (Pressure-Volume-Temperature) correlation tools."""

import json
import math

import pytest

from petro_mcp.tools.pvt import (
    bubble_point,
    calculate_brine_properties,
    calculate_gas_z_factor,
    calculate_oil_compressibility,
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
        assert "Vasquez" in oil["bubble_point_correlation"]

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
# Oil compressibility
# ===========================================================================

class TestOilCompressibility:
    """Tests for calculate_oil_compressibility."""

    def test_above_bubble_point(self):
        """Undersaturated regime — Vasquez-Beggs."""
        result = json.loads(calculate_oil_compressibility(
            api_gravity=35, gas_sg=0.65, temperature=200, pressure=5000,
            bubble_point_pressure=2500, rs_at_pb=400,
        ))
        co = result["oil_compressibility_1_psi"]
        assert co > 0
        assert result["regime"] == "undersaturated"
        assert "Vasquez" in result["correlation"]

    def test_below_bubble_point(self):
        """Saturated regime — material balance."""
        result = json.loads(calculate_oil_compressibility(
            api_gravity=35, gas_sg=0.65, temperature=200, pressure=1500,
            bubble_point_pressure=2500, rs_at_pb=400,
        ))
        co = result["oil_compressibility_1_psi"]
        assert co > 0
        assert result["regime"] == "saturated"

    def test_auto_estimate_pb(self):
        """When Pb and Rs not provided, they are auto-estimated."""
        result = json.loads(calculate_oil_compressibility(
            api_gravity=35, gas_sg=0.65, temperature=200, pressure=3000,
        ))
        assert result["oil_compressibility_1_psi"] > 0
        assert result["inputs"]["bubble_point_pressure_psi"] > 0
        assert result["inputs"]["rs_at_pb_scf_stb"] > 0

    def test_typical_magnitude(self):
        """Oil compressibility typically 5e-6 to 100e-6 1/psi."""
        result = json.loads(calculate_oil_compressibility(
            api_gravity=35, gas_sg=0.65, temperature=200, pressure=4000,
            bubble_point_pressure=2500, rs_at_pb=400,
        ))
        co = result["oil_compressibility_1_psi"]
        assert 1e-7 <= co <= 1e-3

    def test_invalid_inputs(self):
        with pytest.raises(ValueError, match="API gravity must be positive"):
            calculate_oil_compressibility(
                api_gravity=0, gas_sg=0.65, temperature=200, pressure=3000,
            )

    def test_correlation_key_present(self):
        result = json.loads(calculate_oil_compressibility(
            api_gravity=35, gas_sg=0.65, temperature=200, pressure=5000,
            bubble_point_pressure=2500, rs_at_pb=400,
        ))
        assert "correlation" in result


# ===========================================================================
# Brine properties
# ===========================================================================

class TestBrineProperties:
    """Tests for calculate_brine_properties."""

    def test_fresh_water(self):
        """Fresh water (0 ppm salinity)."""
        result = json.loads(calculate_brine_properties(
            temperature=200, pressure=3000, salinity=0,
        ))
        # Fresh water density ~ 62 lb/ft³ at surface, slightly more at 3000 psi
        assert 55 <= result["brine_density_lb_ft3"] <= 70
        assert result["brine_viscosity_cp"] > 0
        assert result["brine_fvf_bbl_stb"] > 0.9
        assert result["brine_compressibility_1_psi"] > 0

    def test_typical_brine(self):
        """Typical formation brine (100,000 ppm)."""
        result = json.loads(calculate_brine_properties(
            temperature=200, pressure=3000, salinity=100000,
        ))
        # Brine denser than fresh water
        fresh = json.loads(calculate_brine_properties(
            temperature=200, pressure=3000, salinity=0,
        ))
        assert result["brine_density_lb_ft3"] > fresh["brine_density_lb_ft3"]

    def test_high_salinity(self):
        """High salinity brine (250,000 ppm)."""
        result = json.loads(calculate_brine_properties(
            temperature=200, pressure=3000, salinity=250000,
        ))
        assert result["brine_density_lb_ft3"] > 60
        assert result["brine_viscosity_cp"] > 0

    def test_salinity_increases_viscosity(self):
        """Higher salinity should increase viscosity."""
        low_s = json.loads(calculate_brine_properties(
            temperature=200, pressure=3000, salinity=10000,
        ))
        high_s = json.loads(calculate_brine_properties(
            temperature=200, pressure=3000, salinity=200000,
        ))
        assert high_s["brine_viscosity_cp"] > low_s["brine_viscosity_cp"]

    def test_temperature_reduces_viscosity(self):
        """Higher temperature should reduce viscosity."""
        cold = json.loads(calculate_brine_properties(
            temperature=100, pressure=3000, salinity=50000,
        ))
        hot = json.loads(calculate_brine_properties(
            temperature=300, pressure=3000, salinity=50000,
        ))
        assert cold["brine_viscosity_cp"] > hot["brine_viscosity_cp"]

    def test_correlations_cited(self):
        result = json.loads(calculate_brine_properties(
            temperature=200, pressure=3000, salinity=50000,
        ))
        assert "McCain" in result["correlations"]["density"]
        assert "McCain" in result["correlations"]["viscosity"]
        assert "McCain" in result["correlations"]["fvf"]
        assert "Osif" in result["correlations"]["compressibility"]

    def test_inputs_echoed(self):
        result = json.loads(calculate_brine_properties(
            temperature=200, pressure=3000, salinity=50000,
        ))
        assert result["inputs"]["temperature_F"] == 200
        assert result["inputs"]["pressure_psi"] == 3000
        assert result["inputs"]["salinity_ppm"] == 50000

    def test_units_present(self):
        result = json.loads(calculate_brine_properties(
            temperature=200, pressure=3000, salinity=50000,
        ))
        assert result["units"]["density"] == "lb/ft³"
        assert result["units"]["viscosity"] == "cp"
        assert result["units"]["fvf"] == "bbl/STB"
        assert result["units"]["compressibility"] == "1/psi"


class TestBrineInvalidInputs:
    def test_negative_temperature(self):
        with pytest.raises(ValueError, match="Temperature must be positive"):
            calculate_brine_properties(temperature=-50, pressure=3000, salinity=0)

    def test_negative_pressure(self):
        with pytest.raises(ValueError, match="Pressure must be positive"):
            calculate_brine_properties(temperature=200, pressure=-100, salinity=0)

    def test_negative_salinity(self):
        with pytest.raises(ValueError, match="Salinity must be non-negative"):
            calculate_brine_properties(temperature=200, pressure=3000, salinity=-100)

    def test_excessive_salinity(self):
        with pytest.raises(ValueError, match="Salinity must be <= 300,000"):
            calculate_brine_properties(temperature=200, pressure=3000, salinity=400000)


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
        assert 0.2 <= result["z_factor"] <= 1.5
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


# ===========================================================================
# Physical consistency checks
# ===========================================================================

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

    def test_vasquez_beggs_rs_increases_with_pressure(self):
        """Vasquez-Beggs Rs should increase with pressure."""
        result_low = json.loads(calculate_pvt(
            api_gravity=35, gas_sg=0.65, temperature=200, pressure=500,
            correlation="vasquez_beggs",
        ))
        result_high = json.loads(calculate_pvt(
            api_gravity=35, gas_sg=0.65, temperature=200, pressure=2000,
            correlation="vasquez_beggs",
        ))
        assert (result_high["oil_properties"]["solution_gor_scf_stb"]
                > result_low["oil_properties"]["solution_gor_scf_stb"])

    def test_petrosky_farshad_bo_gt_1(self):
        """Petrosky-Farshad Bo should always be > 1."""
        for p in [500, 1500, 3000, 5000]:
            result = json.loads(calculate_pvt(
                api_gravity=35, gas_sg=0.70, temperature=200, pressure=p,
                correlation="petrosky_farshad",
            ))
            assert result["oil_properties"]["oil_fvf_bbl_stb"] >= 1.0

    def test_brine_density_increases_with_salinity(self):
        """Brine density should increase with salinity."""
        fresh = json.loads(calculate_brine_properties(
            temperature=200, pressure=3000, salinity=0,
        ))
        saline = json.loads(calculate_brine_properties(
            temperature=200, pressure=3000, salinity=200000,
        ))
        assert saline["brine_density_lb_ft3"] > fresh["brine_density_lb_ft3"]

    def test_brine_bw_near_one(self):
        """Brine FVF should be close to 1.0 (water barely compresses)."""
        result = json.loads(calculate_brine_properties(
            temperature=200, pressure=3000, salinity=50000,
        ))
        assert 0.9 <= result["brine_fvf_bbl_stb"] <= 1.15


# ===========================================================================
# Al-Marhoun bubble point (tested via internal, exposed through compressibility)
# ===========================================================================

class TestAlMarhounPb:
    """Indirect test of Al-Marhoun Pb via internal use."""

    def test_al_marhoun_gives_reasonable_pb(self):
        """Al-Marhoun Pb should be in reasonable range for Middle East oil."""
        from petro_mcp.tools.pvt import _al_marhoun_pb
        pb = _al_marhoun_pb(rs=500, temperature=200, api_gravity=30, gas_sg=0.75)
        assert 500 <= pb <= 6000

    def test_al_marhoun_zero_rs(self):
        from petro_mcp.tools.pvt import _al_marhoun_pb
        pb = _al_marhoun_pb(rs=0, temperature=200, api_gravity=30, gas_sg=0.75)
        assert pb == 14.7

    def test_al_marhoun_increases_with_rs(self):
        from petro_mcp.tools.pvt import _al_marhoun_pb
        pb_low = _al_marhoun_pb(rs=200, temperature=200, api_gravity=30, gas_sg=0.75)
        pb_high = _al_marhoun_pb(rs=800, temperature=200, api_gravity=30, gas_sg=0.75)
        assert pb_high > pb_low
