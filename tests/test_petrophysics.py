"""Tests for petrophysics interpretation tools."""

import json
import math

import pytest

from petro_mcp.tools.petrophysics import (
    calculate_archie_sw,
    calculate_density_porosity,
    calculate_net_pay,
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

# -----------------------------------------------------------------------
# Neutron-density porosity
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Effective porosity
# -----------------------------------------------------------------------

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

# -----------------------------------------------------------------------
# Indonesian Sw
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Permeability — Timur
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Permeability — Coates
# -----------------------------------------------------------------------

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

