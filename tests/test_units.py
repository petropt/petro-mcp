"""Tests for oilfield unit conversion tools."""

import json

import pytest

from petro_mcp.tools.units import convert_units, list_units


def _result(value, from_unit, to_unit):
    """Helper: parse the JSON result and return the 'result' field."""
    return json.loads(convert_units(value, from_unit, to_unit))["result"]


# ---- Volume ----

class TestVolumeConversions:
    def test_bbl_to_m3(self):
        assert abs(_result(1, "bbl", "m3") - 0.158987294928) < 1e-6

    def test_bbl_to_gal(self):
        assert _result(1, "bbl", "gal") == 42.0

    def test_bbl_to_liters(self):
        assert abs(_result(1, "bbl", "liters") - 158.987294928) < 1e-4

    def test_mcf_to_m3(self):
        assert abs(_result(1, "Mcf", "m3") - 28.316846592) < 1e-4

    def test_mmcf_to_m3(self):
        assert abs(_result(1, "MMcf", "m3") - 28316.846592) < 1e-2

    def test_bcf_to_m3(self):
        assert abs(_result(1, "Bcf", "m3") - 28316846.592) < 1


# ---- Rate ----

class TestRateConversions:
    def test_bbl_day_to_m3_day(self):
        assert abs(_result(100, "bbl/day", "m3/day") - 15.8987294928) < 1e-4

    def test_mcf_day_to_m3_day(self):
        assert abs(_result(10, "Mcf/day", "m3/day") - 283.16846592) < 1e-3

    def test_bbl_day_to_bbl_month(self):
        assert abs(_result(100, "bbl/day", "bbl/month") - 3044.0) < 1


# ---- Pressure ----

class TestPressureConversions:
    def test_psi_to_kpa(self):
        assert abs(_result(1000, "psi", "kPa") - 6894.757293168) < 0.01

    def test_psi_to_mpa(self):
        assert abs(_result(1000, "psi", "MPa") - 6.894757293168) < 1e-5

    def test_psi_to_bar(self):
        assert abs(_result(14.696, "psi", "bar") - 1.01325) < 0.001

    def test_psi_to_atm(self):
        assert abs(_result(14.696, "psi", "atm") - 1.0) < 0.001


# ---- Length ----

class TestLengthConversions:
    def test_ft_to_m(self):
        assert abs(_result(1, "ft", "m") - 0.3048) < 1e-6

    def test_in_to_cm(self):
        assert abs(_result(1, "in", "cm") - 2.54) < 1e-6

    def test_miles_to_km(self):
        assert abs(_result(1, "miles", "km") - 1.609344) < 1e-6


# ---- Density ----

class TestDensityConversions:
    def test_gcc_to_kgm3(self):
        assert _result(1.0, "g/cc", "kg/m3") == 1000.0

    def test_gcc_to_lbft3(self):
        assert abs(_result(1.0, "g/cc", "lb/ft3") - 62.428) < 0.01

    def test_api_to_sg(self):
        # 10 API = SG 1.0 (water)
        assert abs(_result(10, "API", "SG") - 1.0) < 1e-6

    def test_sg_to_api(self):
        # SG 0.876 -> ~30 API (light crude)
        api = _result(0.876, "SG", "API")
        assert abs(api - 30.0) < 0.3  # within rounding

    def test_api_to_sg_light_crude(self):
        # 40 API crude
        sg = _result(40, "API", "SG")
        expected = 141.5 / (40 + 131.5)
        assert abs(sg - expected) < 1e-6

    def test_api_sg_roundtrip(self):
        """API -> SG -> API must recover original."""
        original_api = 35.0
        sg = _result(original_api, "API", "SG")
        recovered_api = _result(sg, "SG", "API")
        assert abs(recovered_api - original_api) < 1e-6


# ---- Temperature ----

class TestTemperatureConversions:
    def test_f_to_c_freezing(self):
        assert abs(_result(32, "F", "C") - 0.0) < 1e-6

    def test_f_to_c_boiling(self):
        assert abs(_result(212, "F", "C") - 100.0) < 1e-6

    def test_c_to_f(self):
        assert abs(_result(100, "C", "F") - 212.0) < 1e-6

    def test_f_to_k(self):
        assert abs(_result(32, "F", "K") - 273.15) < 1e-4

    def test_c_to_k(self):
        assert abs(_result(0, "C", "K") - 273.15) < 1e-6

    def test_k_to_c(self):
        assert abs(_result(273.15, "K", "C") - 0.0) < 1e-6

    def test_temperature_roundtrip(self):
        """F -> C -> K -> F must recover original."""
        orig = 150.0
        c = _result(orig, "F", "C")
        k = _result(c, "C", "K")
        back = _result(k, "K", "F")
        assert abs(back - orig) < 1e-6


# ---- Permeability ----

class TestPermeabilityConversions:
    def test_md_to_m2(self):
        assert abs(_result(1, "md", "m2") - 9.869233e-16) < 1e-22

    def test_m2_to_md(self):
        assert abs(_result(9.869233e-16, "m2", "md") - 1.0) < 1e-6


# ---- Viscosity ----

class TestViscosityConversions:
    def test_cp_to_pas(self):
        assert abs(_result(1, "cp", "Pa.s") - 0.001) < 1e-9

    def test_pas_to_cp(self):
        assert abs(_result(0.001, "Pa.s", "cp") - 1.0) < 1e-6


# ---- Energy / BOE ----

class TestEnergyConversions:
    def test_boe_to_mmbtu(self):
        assert abs(_result(1, "BOE", "MMBtu") - 5.8) < 1e-6

    def test_boe_to_mcf_gas(self):
        assert abs(_result(1, "BOE", "Mcf_gas") - 6.0) < 1e-6

    def test_mcf_gas_to_boe(self):
        assert abs(_result(6, "Mcf_gas", "BOE") - 1.0) < 1e-6

    def test_mmbtu_to_mcf_gas(self):
        result = _result(5.8, "MMBtu", "Mcf_gas")
        assert abs(result - 6.0) < 1e-6


# ---- Error handling ----

class TestErrorHandling:
    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown unit"):
            convert_units(1, "foobar", "m3")

    def test_no_conversion_path_raises(self):
        with pytest.raises(ValueError, match="No conversion"):
            convert_units(1, "bbl", "psi")

    def test_invalid_to_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown unit"):
            convert_units(1, "bbl", "wigglefeet")


# ---- Round-trip conversions ----

class TestRoundTrips:
    @pytest.mark.parametrize("from_u,to_u,value", [
        ("bbl", "m3", 123.456),
        ("psi", "kPa", 5000.0),
        ("ft", "m", 10000.0),
        ("g/cc", "lb/ft3", 2.65),
        ("md", "m2", 250.0),
        ("cp", "Pa.s", 0.7),
        ("BOE", "MMBtu", 42.0),
        ("bbl/day", "m3/day", 1500.0),
        ("miles", "km", 3.5),
        ("Mcf", "m3", 100.0),
    ])
    def test_roundtrip(self, from_u, to_u, value):
        """Convert A -> B -> A and verify we recover the original value."""
        intermediate = _result(value, from_u, to_u)
        recovered = _result(intermediate, to_u, from_u)
        assert abs(recovered - value) < abs(value) * 1e-8 + 1e-10


# ---- list_units ----

class TestListUnits:
    def test_returns_all_categories(self):
        data = json.loads(list_units())
        expected = {
            "volume", "rate", "pressure", "length", "density",
            "temperature", "permeability", "viscosity", "energy",
        }
        assert set(data.keys()) == expected

    def test_each_category_has_units(self):
        data = json.loads(list_units())
        for cat, units in data.items():
            assert isinstance(units, list)
            assert len(units) >= 2, f"Category {cat} should have at least 2 units"


# ---- JSON output structure ----

class TestOutputStructure:
    def test_output_keys(self):
        data = json.loads(convert_units(100, "bbl", "m3"))
        assert "value" in data
        assert "from_unit" in data
        assert "to_unit" in data
        assert "result" in data
        assert "conversion_factor" in data

    def test_linear_has_factor(self):
        data = json.loads(convert_units(100, "bbl", "m3"))
        assert data["conversion_factor"] is not None

    def test_nonlinear_temperature_no_factor(self):
        data = json.loads(convert_units(100, "F", "C"))
        assert data["conversion_factor"] is None

    def test_nonlinear_api_no_factor(self):
        data = json.loads(convert_units(30, "API", "SG"))
        assert data["conversion_factor"] is None
