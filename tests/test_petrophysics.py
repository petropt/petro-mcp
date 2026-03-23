"""Tests for petrophysics interpretation tools."""

import json
import math

import pytest

from petro_mcp.tools.petrophysics import (
    calculate_archie_sw,
    calculate_density_porosity,
    calculate_effective_porosity,
    calculate_hpt,
    calculate_indonesian_sw,
    calculate_net_pay,
    calculate_neutron_density_porosity,
    calculate_permeability_coates,
    calculate_permeability_timur,
    calculate_simandoux_sw,
    calculate_sonic_porosity,
    calculate_vshale,
)


# -----------------------------------------------------------------------
# Vshale
# -----------------------------------------------------------------------

class TestCalculateVshale:
    def test_linear_midpoint(self):
        result = json.loads(calculate_vshale(75, 30, 120, "linear"))
        assert result["vshale"] == 0.5
        assert result["igr"] == 0.5
        assert result["method"] == "linear"

    def test_linear_at_clean(self):
        result = json.loads(calculate_vshale(30, 30, 120, "linear"))
        assert result["vshale"] == 0.0

    def test_linear_at_shale(self):
        result = json.loads(calculate_vshale(120, 30, 120, "linear"))
        assert result["vshale"] == 1.0

    def test_clamped_below_clean(self):
        result = json.loads(calculate_vshale(20, 30, 120, "linear"))
        assert result["vshale"] == 0.0

    def test_clamped_above_shale(self):
        result = json.loads(calculate_vshale(130, 30, 120, "linear"))
        assert result["vshale"] == 1.0

    def test_larionov_tertiary(self):
        result = json.loads(calculate_vshale(75, 30, 120, "larionov_tertiary"))
        assert 0 <= result["vshale"] <= 1

    def test_larionov_older(self):
        result = json.loads(calculate_vshale(75, 30, 120, "larionov_older"))
        assert 0 <= result["vshale"] <= 1

    def test_clavier(self):
        result = json.loads(calculate_vshale(75, 30, 120, "clavier"))
        assert 0 <= result["vshale"] <= 1

    def test_equal_gr_raises(self):
        with pytest.raises(ValueError, match="must differ"):
            calculate_vshale(75, 100, 100)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            calculate_vshale(75, 30, 120, "bogus")

    def test_case_insensitive_method(self):
        result = json.loads(calculate_vshale(75, 30, 120, "LINEAR"))
        assert result["method"] == "linear"


# -----------------------------------------------------------------------
# Density porosity
# -----------------------------------------------------------------------

class TestCalculateDensityPorosity:
    def test_typical_sandstone(self):
        result = json.loads(calculate_density_porosity(2.3, 2.65, 1.0))
        assert abs(result["density_porosity"] - 0.2121) < 0.001

    def test_zero_porosity(self):
        result = json.loads(calculate_density_porosity(2.65, 2.65, 1.0))
        assert result["density_porosity"] == 0.0

    def test_negative_porosity_clamped(self):
        result = json.loads(calculate_density_porosity(2.8, 2.65, 1.0))
        assert result["density_porosity"] == 0.0

    def test_high_porosity_clamped(self):
        result = json.loads(calculate_density_porosity(0.5, 2.65, 1.0))
        assert result["density_porosity"] == 1.0

    def test_limestone_matrix(self):
        result = json.loads(calculate_density_porosity(2.4, 2.71, 1.0))
        assert 0 < result["density_porosity"] < 0.5

    def test_equal_matrix_fluid_raises(self):
        with pytest.raises(ValueError, match="must differ"):
            calculate_density_porosity(2.3, 1.0, 1.0)

    def test_zero_density_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_density_porosity(0, 2.65, 1.0)


# -----------------------------------------------------------------------
# Sonic porosity
# -----------------------------------------------------------------------

class TestCalculateSonicPorosity:
    def test_wyllie_typical(self):
        result = json.loads(calculate_sonic_porosity(90, 55.5, 189.0, "wyllie"))
        expected = (90 - 55.5) / (189.0 - 55.5)
        assert abs(result["sonic_porosity"] - round(expected, 4)) < 0.001

    def test_raymer_typical(self):
        result = json.loads(calculate_sonic_porosity(90, 55.5, 189.0, "raymer"))
        expected = 0.625 * (90 - 55.5) / 90
        assert abs(result["sonic_porosity"] - round(expected, 4)) < 0.001

    def test_clamped_to_zero(self):
        result = json.loads(calculate_sonic_porosity(50, 55.5, 189.0, "wyllie"))
        assert result["sonic_porosity"] == 0.0

    def test_equal_matrix_fluid_raises(self):
        with pytest.raises(ValueError, match="must differ"):
            calculate_sonic_porosity(90, 55.5, 55.5)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            calculate_sonic_porosity(90, 55.5, 189.0, "bogus")


# -----------------------------------------------------------------------
# Neutron-density porosity
# -----------------------------------------------------------------------

class TestCalculateNeutronDensityPorosity:
    def test_typical_values(self):
        result = json.loads(calculate_neutron_density_porosity(0.25, 0.20))
        expected = math.sqrt((0.25**2 + 0.20**2) / 2.0)
        assert abs(result["neutron_density_porosity"] - round(expected, 4)) < 0.001

    def test_zero_inputs(self):
        result = json.loads(calculate_neutron_density_porosity(0.0, 0.0))
        assert result["neutron_density_porosity"] == 0.0

    def test_clamped_to_one(self):
        result = json.loads(calculate_neutron_density_porosity(1.5, 1.5))
        assert result["neutron_density_porosity"] == 1.0


# -----------------------------------------------------------------------
# Effective porosity
# -----------------------------------------------------------------------

class TestCalculateEffectivePorosity:
    def test_typical(self):
        result = json.loads(calculate_effective_porosity(0.25, 0.2))
        assert result["effective_porosity"] == 0.2

    def test_clean_sand(self):
        result = json.loads(calculate_effective_porosity(0.25, 0.0))
        assert result["effective_porosity"] == 0.25

    def test_pure_shale(self):
        result = json.loads(calculate_effective_porosity(0.25, 1.0))
        assert result["effective_porosity"] == 0.0

    def test_invalid_phi_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            calculate_effective_porosity(1.5, 0.2)

    def test_invalid_vshale_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            calculate_effective_porosity(0.25, 1.5)


# -----------------------------------------------------------------------
# Archie Sw
# -----------------------------------------------------------------------

class TestCalculateArchieSw:
    def test_typical_clean_sand(self):
        result = json.loads(calculate_archie_sw(10.0, 0.20, 0.05))
        # Sw = (1*0.05 / (0.2^2 * 10))^(1/2) = sqrt(0.05/0.4) = sqrt(0.125)
        expected = math.sqrt(0.05 / (0.04 * 10.0))
        assert abs(result["water_saturation"] - round(expected, 4)) < 0.001
        assert abs(result["hydrocarbon_saturation"] - round(1 - expected, 4)) < 0.001

    def test_high_rt_low_sw(self):
        result = json.loads(calculate_archie_sw(10000.0, 0.20, 0.05))
        assert result["water_saturation"] < 0.05

    def test_low_rt_clamped(self):
        result = json.loads(calculate_archie_sw(0.1, 0.20, 0.05))
        assert result["water_saturation"] == 1.0

    def test_zero_porosity_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_archie_sw(10.0, 0.0, 0.05)

    def test_zero_rt_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_archie_sw(0.0, 0.20, 0.05)

    def test_custom_cementation(self):
        result = json.loads(calculate_archie_sw(10.0, 0.20, 0.05, a=0.81, m=2.0, n=2.0))
        assert 0 <= result["water_saturation"] <= 1


# -----------------------------------------------------------------------
# Simandoux Sw
# -----------------------------------------------------------------------

class TestCalculateSimandouxSw:
    def test_reduces_to_archie_when_clean(self):
        sim = json.loads(calculate_simandoux_sw(10.0, 0.20, 0.05, 0.0, 5.0))
        arch = json.loads(calculate_archie_sw(10.0, 0.20, 0.05))
        assert abs(sim["water_saturation"] - arch["water_saturation"]) < 0.001

    def test_shaly_sand_differs_from_archie(self):
        sim = json.loads(calculate_simandoux_sw(10.0, 0.20, 0.05, 0.3, 5.0))
        arch = json.loads(calculate_archie_sw(10.0, 0.20, 0.05))
        # Shale conductivity changes apparent Sw vs Archie
        assert sim["water_saturation"] != arch["water_saturation"]
        assert 0 <= sim["water_saturation"] <= 1

    def test_zero_porosity_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_simandoux_sw(10.0, 0.0, 0.05, 0.3, 5.0)

    def test_zero_rsh_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_simandoux_sw(10.0, 0.20, 0.05, 0.3, 0.0)

    def test_clamped(self):
        result = json.loads(calculate_simandoux_sw(0.1, 0.20, 0.05, 0.3, 5.0))
        assert result["water_saturation"] <= 1.0


# -----------------------------------------------------------------------
# Indonesian Sw
# -----------------------------------------------------------------------

class TestCalculateIndonesianSw:
    def test_reduces_to_archie_when_clean(self):
        indo = json.loads(calculate_indonesian_sw(10.0, 0.20, 0.05, 0.0, 5.0))
        arch = json.loads(calculate_archie_sw(10.0, 0.20, 0.05))
        assert abs(indo["water_saturation"] - arch["water_saturation"]) < 0.001

    def test_typical_shaly_sand(self):
        result = json.loads(calculate_indonesian_sw(10.0, 0.20, 0.05, 0.3, 5.0))
        assert 0 <= result["water_saturation"] <= 1

    def test_clamped(self):
        result = json.loads(calculate_indonesian_sw(0.1, 0.20, 0.05, 0.3, 5.0))
        assert result["water_saturation"] <= 1.0

    def test_zero_porosity_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_indonesian_sw(10.0, 0.0, 0.05, 0.3, 5.0)

    def test_zero_rsh_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_indonesian_sw(10.0, 0.20, 0.05, 0.3, 0.0)


# -----------------------------------------------------------------------
# Permeability — Timur
# -----------------------------------------------------------------------

class TestCalculatePermeabilityTimur:
    def test_typical(self):
        result = json.loads(calculate_permeability_timur(0.20, 0.25))
        # Timur with phi=20%, Swirr=25%: k = 0.136 * 20^4.4 / 25^2
        expected = 0.136 * 20**4.4 / 25**2
        assert abs(result["permeability_md"] - round(expected, 4)) < 1.0

    def test_high_porosity(self):
        result = json.loads(calculate_permeability_timur(0.30, 0.15))
        assert result["permeability_md"] > 100  # high-perm case

    def test_zero_swirr_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_permeability_timur(0.20, 0.0)

    def test_zero_porosity_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_permeability_timur(0.0, 0.25)

    def test_negative_porosity_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_permeability_timur(-0.1, 0.25)


# -----------------------------------------------------------------------
# Permeability — Coates
# -----------------------------------------------------------------------

class TestCalculatePermeabilityCoates:
    def test_typical(self):
        result = json.loads(calculate_permeability_coates(0.20, 0.05, 0.15))
        # Coates with phi=20% (in pct), C=10: k = ((20/10)^2 * (0.15/0.05))^2 = (4*3)^2 = 144
        expected = ((20.0 / 10.0) ** 2 * (0.15 / 0.05)) ** 2
        assert abs(result["permeability_md"] - round(expected, 4)) < 1.0

    def test_zero_ffi_returns_zero(self):
        result = json.loads(calculate_permeability_coates(0.20, 0.05, 0.0))
        assert result["permeability_md"] == 0.0

    def test_zero_bvi_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_permeability_coates(0.20, 0.0, 0.15)

    def test_custom_c(self):
        result = json.loads(calculate_permeability_coates(0.20, 0.05, 0.15, c=8.0))
        assert result["permeability_md"] > 0


# -----------------------------------------------------------------------
# Net pay
# -----------------------------------------------------------------------

class TestCalculateNetPay:
    def _sample_data(self):
        depths = [5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009]
        phi =    [0.15, 0.18, 0.20, 0.03, 0.02, 0.22, 0.19, 0.17, 0.04, 0.16]
        sw =     [0.30, 0.25, 0.35, 0.80, 0.90, 0.20, 0.30, 0.40, 0.85, 0.45]
        vshale = [0.10, 0.15, 0.20, 0.70, 0.80, 0.05, 0.10, 0.25, 0.75, 0.20]
        return depths, phi, sw, vshale

    def test_partial_pay(self):
        depths, phi, sw, vshale = self._sample_data()
        result = json.loads(calculate_net_pay(depths, phi, sw, vshale))
        assert result["net_pay_ft"] > 0
        assert result["net_pay_ft"] < result["gross_thickness_ft"]
        assert 0 < result["net_to_gross"] < 1

    def test_all_pay(self):
        depths = [100, 101, 102, 103, 104]
        phi = [0.20] * 5
        sw = [0.25] * 5
        vshale = [0.10] * 5
        result = json.loads(calculate_net_pay(depths, phi, sw, vshale))
        assert result["net_to_gross"] == 1.0
        assert all(result["pay_flags"])

    def test_no_pay(self):
        depths = [100, 101, 102]
        phi = [0.02] * 3  # below cutoff
        sw = [0.90] * 3
        vshale = [0.80] * 3
        result = json.loads(calculate_net_pay(depths, phi, sw, vshale))
        assert result["net_pay_ft"] == 0
        assert result["net_to_gross"] == 0.0
        assert result["avg_porosity_pay"] is None
        assert "note" in result

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            calculate_net_pay([1, 2, 3], [0.1, 0.2], [0.3, 0.4, 0.5], [0.1, 0.2, 0.3])

    def test_single_point_raises(self):
        with pytest.raises(ValueError, match="At least 2"):
            calculate_net_pay([100], [0.2], [0.3], [0.1])

    def test_pay_flags_correct(self):
        depths = [100, 101, 102]
        phi = [0.20, 0.02, 0.15]
        sw = [0.30, 0.30, 0.30]
        vshale = [0.10, 0.10, 0.10]
        result = json.loads(calculate_net_pay(depths, phi, sw, vshale))
        assert result["pay_flags"] == [True, False, True]

    def test_average_properties(self):
        depths = [100, 101, 102]
        phi = [0.20, 0.02, 0.20]
        sw = [0.30, 0.90, 0.30]
        vshale = [0.10, 0.80, 0.10]
        result = json.loads(calculate_net_pay(depths, phi, sw, vshale))
        # Only pay intervals (indices 0 and 2) contribute to averages
        assert result["avg_porosity_pay"] == 0.2
        assert result["avg_sw_pay"] == 0.3
        assert result["avg_vshale_pay"] == 0.1


# -----------------------------------------------------------------------
# Hydrocarbon pore thickness
# -----------------------------------------------------------------------

class TestCalculateHPT:
    def test_typical(self):
        result = json.loads(calculate_hpt(100, 0.2, 0.3, 0.8))
        expected = 100 * 0.2 * (1 - 0.3) * 0.8
        assert abs(result["hpt_ft"] - round(expected, 4)) < 0.01

    def test_zero_porosity(self):
        result = json.loads(calculate_hpt(100, 0.0, 0.3, 0.8))
        assert result["hpt_ft"] == 0.0

    def test_full_water(self):
        result = json.loads(calculate_hpt(100, 0.2, 1.0, 0.8))
        assert result["hpt_ft"] == 0.0

    def test_negative_thickness_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_hpt(-10, 0.2, 0.3)

    def test_invalid_sw_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            calculate_hpt(100, 0.2, 1.5)
