"""Tests for reservoir engineering material balance tools."""

import json
import math

import pytest

from petro_mcp.tools.reservoir import (
    calculate_pz_analysis,
    calculate_recovery_factor,
    calculate_volumetric_ogip,
    calculate_volumetric_ooip,
)


# ===========================================================================
# P/Z Analysis (Gas Material Balance)
# ===========================================================================

class TestPZAnalysis:
    """Tests for gas material balance P/Z analysis."""

    def test_textbook_example(self):
        """Test with a textbook-style gas reservoir.

        Known: Pi/Zi = 4000 psi, OGIP = 100 Bcf.
        P/Z should decline linearly with Gp.
        At Gp=0, P/Z=4000. At Gp=100, P/Z=0.
        So slope = -40 per Bcf.
        """
        # Generate synthetic data along the line P/Z = 4000 - 40*Gp
        gp = [0, 10, 20, 30, 40, 50]
        pz = [4000, 3600, 3200, 2800, 2400, 2000]

        result = json.loads(calculate_pz_analysis(pz, gp))

        assert result["method"] == "P/Z vs Gp (Gas Material Balance)"
        assert result["results"]["ogip_bcf"] == pytest.approx(100.0, rel=1e-3)
        assert result["results"]["r_squared"] == pytest.approx(1.0, rel=1e-6)
        assert result["results"]["intercept_psi"] == pytest.approx(4000, rel=1e-3)
        assert result["results"]["slope"] == pytest.approx(-40, rel=1e-3)

    def test_recovery_factor(self):
        """Current recovery factor matches Gp_max / OGIP."""
        gp = [0, 25, 50]
        pz = [3000, 2250, 1500]
        # OGIP = 3000/15 = 200... slope = (1500-3000)/(50-0) = -30
        # intercept = 3000, OGIP = 3000/30 = 100

        result = json.loads(calculate_pz_analysis(pz, gp))
        ogip = result["results"]["ogip_bcf"]
        rf = result["results"]["current_recovery_factor"]

        assert ogip == pytest.approx(100.0, rel=1e-3)
        assert rf == pytest.approx(0.5, rel=1e-3)

    def test_abandonment_pressure(self):
        """Abandonment recovery is calculated when abandonment_pressure given."""
        gp = [0, 10, 20, 30, 40, 50]
        pz = [4000, 3600, 3200, 2800, 2400, 2000]

        result = json.loads(calculate_pz_analysis(pz, gp, abandonment_pressure=800))

        assert "abandonment" in result
        abn = result["abandonment"]
        assert abn["abandonment_pressure_psi"] == 800
        # At P/Z=800: Gp = (4000-800)/40 = 80 Bcf
        assert abn["recoverable_gas_bcf"] == pytest.approx(80.0, rel=1e-3)
        assert abn["abandonment_recovery_factor"] == pytest.approx(0.8, rel=1e-3)

    def test_two_points_minimum(self):
        """Should work with exactly 2 data points."""
        result = json.loads(calculate_pz_analysis([3000, 2000], [0, 20]))
        assert result["results"]["ogip_bcf"] is not None

    def test_single_point_raises(self):
        """Single point is insufficient for regression."""
        with pytest.raises(ValueError, match="At least 2"):
            calculate_pz_analysis([3000], [0])

    def test_empty_arrays_raises(self):
        """Empty arrays should raise."""
        with pytest.raises(ValueError, match="At least 2"):
            calculate_pz_analysis([], [])

    def test_mismatched_lengths_raises(self):
        """Mismatched array lengths should raise."""
        with pytest.raises(ValueError, match="equal length"):
            calculate_pz_analysis([3000, 2000], [0, 10, 20])

    def test_negative_pressure_raises(self):
        """Negative pressures should raise."""
        with pytest.raises(ValueError, match="positive"):
            calculate_pz_analysis([-100, 2000], [0, 10])

    def test_negative_cumulative_gas_raises(self):
        """Negative cumulative gas should raise."""
        with pytest.raises(ValueError, match="non-negative"):
            calculate_pz_analysis([3000, 2000], [-5, 10])

    def test_noisy_data(self):
        """P/Z analysis should handle noisy data with R² < 1."""
        gp = [0, 10, 20, 30, 40, 50]
        pz = [4000, 3650, 3150, 2850, 2350, 2050]  # slight noise

        result = json.loads(calculate_pz_analysis(pz, gp))
        assert result["results"]["ogip_bcf"] is not None
        assert 0.95 < result["results"]["r_squared"] < 1.0


# ===========================================================================
# Volumetric OOIP
# ===========================================================================

class TestVolumetricOOIP:
    """Tests for volumetric OOIP calculation."""

    def test_hand_calculation(self):
        """OOIP = 7758 * 640 * 50 * 0.2 * (1-0.25) / 1.2 = 31,032,000 STB."""
        result = json.loads(calculate_volumetric_ooip(
            area_acres=640,
            thickness_ft=50,
            porosity=0.2,
            sw=0.25,
            bo=1.2,
        ))

        expected = 7758 * 640 * 50 * 0.2 * 0.75 / 1.2
        assert result["results"]["ooip_stb"] == pytest.approx(expected, rel=1e-6)

    def test_unit_area(self):
        """1 acre, 1 ft, phi=1, Sw=0, Bo=1 should give 7758 STB."""
        result = json.loads(calculate_volumetric_ooip(1, 1, 1.0, 0.0, 1.0))
        assert result["results"]["ooip_stb"] == pytest.approx(7758.0, rel=1e-6)

    def test_mmstb_conversion(self):
        """Check that MMSTB = STB / 1e6."""
        result = json.loads(calculate_volumetric_ooip(640, 50, 0.2, 0.25, 1.2))
        stb = result["results"]["ooip_stb"]
        mmstb = result["results"]["ooip_mmstb"]
        assert mmstb == pytest.approx(stb / 1e6, rel=1e-4)

    def test_zero_porosity(self):
        """Zero porosity should give zero OOIP (but fraction validator allows 0)."""
        result = json.loads(calculate_volumetric_ooip(640, 50, 0.0, 0.25, 1.2))
        assert result["results"]["ooip_stb"] == 0.0

    def test_negative_area_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_volumetric_ooip(-100, 50, 0.2, 0.25, 1.2)

    def test_porosity_out_of_range_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            calculate_volumetric_ooip(640, 50, 1.5, 0.25, 1.2)

    def test_sw_out_of_range_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            calculate_volumetric_ooip(640, 50, 0.2, -0.1, 1.2)


# ===========================================================================
# Volumetric OGIP
# ===========================================================================

class TestVolumetricOGIP:
    """Tests for volumetric OGIP calculation."""

    def test_hand_calculation(self):
        """OGIP = 43560 * 640 * 100 * 0.15 * (1-0.3) / 0.005."""
        result = json.loads(calculate_volumetric_ogip(
            area_acres=640,
            thickness_ft=100,
            porosity=0.15,
            sw=0.3,
            bg=0.005,
        ))

        expected = 43560 * 640 * 100 * 0.15 * 0.70 / 0.005
        assert result["results"]["ogip_scf"] == pytest.approx(expected, rel=1e-6)

    def test_bcf_conversion(self):
        """Check Bcf = scf / 1e9."""
        result = json.loads(calculate_volumetric_ogip(640, 100, 0.15, 0.3, 0.005))
        scf = result["results"]["ogip_scf"]
        bcf = result["results"]["ogip_bcf"]
        assert bcf == pytest.approx(scf / 1e9, rel=1e-4)

    def test_negative_bg_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_volumetric_ogip(640, 100, 0.15, 0.3, -0.005)


# ===========================================================================
# Recovery Factor
# ===========================================================================

class TestRecoveryFactor:
    """Tests for recovery factor calculation."""

    def test_simple_rf(self):
        """RF = 2e6 / 10e6 = 0.2 (20%)."""
        result = json.loads(calculate_recovery_factor(10e6, 2e6))
        assert result["results"]["recovery_factor"] == pytest.approx(0.2, rel=1e-6)
        assert result["results"]["recovery_factor_pct"] == pytest.approx(20.0, rel=1e-4)

    def test_zero_production(self):
        """RF = 0 when no production."""
        result = json.loads(calculate_recovery_factor(10e6, 0))
        assert result["results"]["recovery_factor"] == 0.0

    def test_full_recovery(self):
        """RF = 1.0 at 100% recovery."""
        result = json.loads(calculate_recovery_factor(1000, 1000))
        assert result["results"]["recovery_factor"] == pytest.approx(1.0, rel=1e-6)

    def test_remaining_reserves(self):
        """Remaining = OOIP - Np."""
        result = json.loads(calculate_recovery_factor(10e6, 3e6))
        assert result["results"]["remaining_in_place"] == pytest.approx(7e6, rel=1e-6)

    def test_zero_ooip_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_recovery_factor(0, 100)

    def test_negative_production_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            calculate_recovery_factor(10e6, -100)
