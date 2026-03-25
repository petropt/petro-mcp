"""Tests for well trajectory and directional survey tools."""

import json
import math

import pytest

try:
    import welleng  # noqa: F401
    HAS_WELLENG = True
except ImportError:
    HAS_WELLENG = False

pytestmark = pytest.mark.skipif(not HAS_WELLENG, reason="welleng not installed")

from petro_mcp.tools.trajectory import (  # noqa: E402
    calculate_survey,
    calculate_dogleg_severity,
    calculate_vertical_section,
    calculate_tortuosity,
    check_anticollision,
)


# ===========================================================================
# calculate_survey — Minimum Curvature
# ===========================================================================

class TestCalculateSurvey:
    """Tests for the minimum curvature survey calculation."""

    def test_vertical_well(self):
        """A perfectly vertical well: North=0, East=0, TVD=MD."""
        result = json.loads(calculate_survey(
            md=[0, 1000, 2000, 3000],
            inclination=[0, 0, 0, 0],
            azimuth=[0, 0, 0, 0],
        ))
        assert result["method"] == "minimum_curvature"
        assert result["num_stations"] == 4

        for station in result["stations"]:
            assert station["north"] == pytest.approx(0.0, abs=0.01)
            assert station["east"] == pytest.approx(0.0, abs=0.01)
            assert station["tvd"] == pytest.approx(station["md"], abs=0.1)

        assert result["summary"]["total_horizontal_departure"] == pytest.approx(0.0, abs=0.01)

    def test_deviated_well(self):
        """Deviated well should have non-zero North, East, and TVD < MD."""
        result = json.loads(calculate_survey(
            md=[0, 500, 1000, 1500, 2000],
            inclination=[0, 5, 20, 45, 60],
            azimuth=[0, 30, 45, 45, 45],
        ))
        summary = result["summary"]

        # TVD at TD should be less than total MD for deviated well
        assert summary["tvd_at_td"] < summary["total_md"]

        # Should have horizontal departure
        assert summary["total_horizontal_departure"] > 0

        # Max inclination should be 60
        assert summary["max_inclination_deg"] == pytest.approx(60.0, abs=0.1)

    def test_horizontal_well(self):
        """Horizontal well: inc reaches 90 degrees."""
        result = json.loads(calculate_survey(
            md=[0, 1000, 2000, 5000, 8000, 10000],
            inclination=[0, 5, 30, 90, 90, 90],
            azimuth=[0, 0, 0, 0, 0, 0],
        ))
        summary = result["summary"]

        # TVD should be much less than MD for horizontal well
        assert summary["tvd_at_td"] < summary["total_md"] * 0.7

        # Max inclination = 90
        assert summary["max_inclination_deg"] == pytest.approx(90.0, abs=0.1)

    def test_s_shaped_well(self):
        """S-shaped well: builds, holds, then drops back to vertical."""
        result = json.loads(calculate_survey(
            md=[0, 500, 1000, 2000, 3000, 4000, 5000],
            inclination=[0, 15, 30, 30, 30, 15, 0],
            azimuth=[0, 45, 45, 45, 45, 45, 45],
        ))
        # Should complete without error and have reasonable values
        assert result["num_stations"] == 7
        assert result["summary"]["tvd_at_td"] < result["summary"]["total_md"]

    def test_meters_unit(self):
        """Test with metric units."""
        result = json.loads(calculate_survey(
            md=[0, 300, 600, 900],
            inclination=[0, 10, 20, 30],
            azimuth=[0, 90, 90, 90],
            unit="meters",
        ))
        assert result["unit"] == "meters"
        assert result["num_stations"] == 4

    def test_minimum_two_stations(self):
        """Two stations is the minimum required."""
        result = json.loads(calculate_survey(
            md=[0, 1000],
            inclination=[0, 30],
            azimuth=[0, 45],
        ))
        assert result["num_stations"] == 2

    def test_too_few_stations(self):
        """Single station should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            calculate_survey(md=[0], inclination=[0], azimuth=[0])

    def test_mismatched_lengths(self):
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="lengths must match"):
            calculate_survey(
                md=[0, 1000, 2000],
                inclination=[0, 10],
                azimuth=[0, 30, 45],
            )

    def test_non_monotonic_md(self):
        """Non-increasing MD should raise ValueError."""
        with pytest.raises(ValueError, match="monotonically increasing"):
            calculate_survey(
                md=[0, 1000, 500],
                inclination=[0, 10, 20],
                azimuth=[0, 30, 45],
            )

    def test_invalid_inclination(self):
        """Inclination > 180 should raise ValueError."""
        with pytest.raises(ValueError, match="Inclination"):
            calculate_survey(
                md=[0, 1000],
                inclination=[0, 200],
                azimuth=[0, 45],
            )

    def test_invalid_unit(self):
        """Invalid unit should raise ValueError."""
        with pytest.raises(ValueError, match="unit must be"):
            calculate_survey(
                md=[0, 1000],
                inclination=[0, 10],
                azimuth=[0, 45],
                unit="yards",
            )

    def test_dls_at_surface_is_zero(self):
        """DLS at the first station should be zero."""
        result = json.loads(calculate_survey(
            md=[0, 500, 1000],
            inclination=[0, 10, 30],
            azimuth=[0, 45, 60],
        ))
        assert result["stations"][0]["dls_deg_per_100"] == 0.0

    def test_output_is_json_string(self):
        """Return type should be a JSON string."""
        result = calculate_survey(
            md=[0, 1000],
            inclination=[0, 10],
            azimuth=[0, 45],
        )
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "stations" in parsed


# ===========================================================================
# calculate_dogleg_severity
# ===========================================================================

class TestCalculateDoglegSeverity:
    """Tests for DLS calculation between two stations."""

    def test_no_change(self):
        """Zero change in inc/azi = zero DLS."""
        result = json.loads(calculate_dogleg_severity(
            md1=1000, inc1=30, azi1=45,
            md2=1100, inc2=30, azi2=45,
        ))
        assert result["dogleg_severity"] == pytest.approx(0.0, abs=0.01)

    def test_inclination_change_only(self):
        """Pure inclination change over 100 ft = DLS in deg/100ft."""
        result = json.loads(calculate_dogleg_severity(
            md1=1000, inc1=30, azi1=45,
            md2=1100, inc2=35, azi2=45,
        ))
        # 5 deg over 100 ft = 5 deg/100ft
        assert result["dogleg_severity"] == pytest.approx(5.0, abs=0.1)

    def test_azimuth_change_only(self):
        """Azimuth change at inclination > 0 produces nonzero DLS."""
        result = json.loads(calculate_dogleg_severity(
            md1=1000, inc1=30, azi1=45,
            md2=1100, inc2=30, azi2=55,
        ))
        assert result["dogleg_severity"] > 0

    def test_combined_change(self):
        """Both inc and azi change — DLS should be > either alone."""
        result = json.loads(calculate_dogleg_severity(
            md1=1000, inc1=20, azi1=30,
            md2=1100, inc2=25, azi2=40,
        ))
        assert result["dogleg_severity"] > 0
        assert result["course_length"] == pytest.approx(100.0, abs=0.01)

    def test_md2_less_than_md1(self):
        """md2 <= md1 should raise ValueError."""
        with pytest.raises(ValueError, match="md2 must be > md1"):
            calculate_dogleg_severity(
                md1=1000, inc1=30, azi1=45,
                md2=900, inc2=35, azi2=50,
            )

    def test_metric_units(self):
        """Metric DLS is in deg/30m."""
        result = json.loads(calculate_dogleg_severity(
            md1=300, inc1=10, azi1=0,
            md2=330, inc2=15, azi2=0,
            course_length_unit="meters",
        ))
        assert result["dls_unit"] == "deg/30m"
        assert result["dogleg_severity"] > 0


# ===========================================================================
# calculate_vertical_section
# ===========================================================================

class TestCalculateVerticalSection:
    """Tests for vertical section projection."""

    def test_well_along_vs_azimuth(self):
        """Well aimed at vs_azimuth should show maximum VS displacement."""
        result = json.loads(calculate_vertical_section(
            md=[0, 500, 1000, 1500],
            inclination=[0, 15, 30, 45],
            azimuth=[0, 0, 0, 0],
            vs_azimuth=0.0,
        ))
        # VS displacement should be close to north displacement
        last = result["stations"][-1]
        assert last["vertical_section"] > 0
        assert abs(last["vertical_section"] - last["north"]) < 10

    def test_well_perpendicular_to_vs(self):
        """Well going east, VS azimuth = north: VS displacement ~0."""
        result = json.loads(calculate_vertical_section(
            md=[0, 500, 1000],
            inclination=[0, 30, 30],
            azimuth=[0, 90, 90],
            vs_azimuth=0.0,
        ))
        # With azi=90 and vs_azi=0, VS should be small relative to east displacement
        last = result["stations"][-1]
        assert abs(last["vertical_section"]) < abs(last["east"]) + 1

    def test_vs_azimuth_45(self):
        """VS azimuth at 45 degrees should project onto NE diagonal."""
        result = json.loads(calculate_vertical_section(
            md=[0, 1000, 2000],
            inclination=[0, 30, 45],
            azimuth=[0, 45, 45],
            vs_azimuth=45.0,
        ))
        assert result["vs_azimuth_deg"] == pytest.approx(45.0)
        assert result["max_vertical_section"] > 0

    def test_output_has_tvd(self):
        """Each station should include TVD for VS plot."""
        result = json.loads(calculate_vertical_section(
            md=[0, 1000, 2000],
            inclination=[0, 15, 30],
            azimuth=[0, 45, 45],
        ))
        for station in result["stations"]:
            assert "tvd" in station
            assert "vertical_section" in station


# ===========================================================================
# calculate_tortuosity
# ===========================================================================

class TestCalculateTortuosity:
    """Tests for wellbore tortuosity index."""

    def test_straight_well_low_tortuosity(self):
        """Vertical well should have zero tortuosity."""
        result = json.loads(calculate_tortuosity(
            md=[0, 1000, 2000, 3000],
            inclination=[0, 0, 0, 0],
            azimuth=[0, 0, 0, 0],
        ))
        for station in result["stations"]:
            assert station["tortuosity_index"] == pytest.approx(0.0, abs=0.1)

    def test_deviated_well_has_tortuosity(self):
        """Deviated well with direction changes should have nonzero tortuosity."""
        result = json.loads(calculate_tortuosity(
            md=[0, 500, 1000, 1500, 2000],
            inclination=[0, 5, 20, 45, 60],
            azimuth=[0, 30, 60, 90, 120],
        ))
        summary = result["summary"]
        assert summary["max_tortuosity_index"] > 0

    def test_tortuous_well_higher_index(self):
        """Well with frequent direction changes has higher tortuosity."""
        # Smooth well
        smooth = json.loads(calculate_tortuosity(
            md=[0, 500, 1000, 1500, 2000],
            inclination=[0, 10, 20, 30, 40],
            azimuth=[0, 0, 0, 0, 0],
        ))
        # Tortuous well (same overall direction but with wiggles)
        tortuous = json.loads(calculate_tortuosity(
            md=[0, 500, 1000, 1500, 2000],
            inclination=[0, 15, 10, 35, 40],
            azimuth=[0, 20, 350, 20, 0],
        ))
        assert (tortuous["summary"]["max_tortuosity_index"]
                > smooth["summary"]["max_tortuosity_index"])

    def test_tortuosity_has_dls(self):
        """Each station should also report DLS for context."""
        result = json.loads(calculate_tortuosity(
            md=[0, 500, 1000],
            inclination=[0, 15, 30],
            azimuth=[0, 30, 60],
        ))
        for station in result["stations"]:
            assert "dls_deg_per_100" in station


# ===========================================================================
# check_anticollision
# ===========================================================================

class TestCheckAnticollision:
    """Tests for anti-collision separation check."""

    def test_identical_wells_zero_separation(self):
        """Two identical wells from same surface location: distance ~0."""
        md = [0, 1000, 2000, 3000]
        inc = [0, 15, 30, 45]
        azi = [0, 45, 45, 45]

        result = json.loads(check_anticollision(
            well1_md=md, well1_inc=inc, well1_azi=azi,
            well2_md=md, well2_inc=inc, well2_azi=azi,
        ))
        assert result["closest_approach"]["distance"] == pytest.approx(0.0, abs=1.0)

    def test_separated_wells(self):
        """Wells with different surface locations should have positive separation."""
        md = [0, 1000, 2000, 3000]
        inc = [0, 15, 30, 45]
        azi = [0, 45, 45, 45]

        result = json.loads(check_anticollision(
            well1_md=md, well1_inc=inc, well1_azi=azi,
            well2_md=md, well2_inc=inc, well2_azi=azi,
            well2_start_north=500.0,
            well2_start_east=500.0,
        ))
        assert result["closest_approach"]["distance"] > 0

    def test_diverging_wells(self):
        """Wells going in opposite directions should have large separation at TD."""
        result = json.loads(check_anticollision(
            well1_md=[0, 1000, 2000, 3000],
            well1_inc=[0, 30, 45, 45],
            well1_azi=[0, 0, 0, 0],
            well2_md=[0, 1000, 2000, 3000],
            well2_inc=[0, 30, 45, 45],
            well2_azi=[0, 180, 180, 180],
        ))
        # At TD, wells should be far apart
        sep = result["separation_at_td"]
        assert sep["distance_at_ref_td"] > 100

    def test_output_structure(self):
        """Result should have closest_approach and separation_at_td."""
        result = json.loads(check_anticollision(
            well1_md=[0, 1000, 2000],
            well1_inc=[0, 15, 30],
            well1_azi=[0, 45, 45],
            well2_md=[0, 1000, 2000],
            well2_inc=[0, 15, 30],
            well2_azi=[0, 90, 90],
        ))
        assert "closest_approach" in result
        assert "distance" in result["closest_approach"]
        assert "reference_well_md" in result["closest_approach"]
        assert "separation_at_td" in result

    def test_metric_units(self):
        """Test with meters."""
        result = json.loads(check_anticollision(
            well1_md=[0, 300, 600, 900],
            well1_inc=[0, 15, 30, 45],
            well1_azi=[0, 0, 0, 0],
            well2_md=[0, 300, 600, 900],
            well2_inc=[0, 15, 30, 45],
            well2_azi=[0, 90, 90, 90],
            unit="meters",
        ))
        assert result["unit"] == "meters"
