"""Tests for production trend analysis and anomaly detection."""

import json
import os

import pytest

from petro_mcp.tools.trends import analyze_production_trends

SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_production.csv")


def _load(file_path=SAMPLE_CSV, well_name=None):
    return json.loads(analyze_production_trends(file_path, well_name))


# --- Shut-in detection ---


def test_shutin_detected_wolfcamp():
    """Wolfcamp A-1H has a shut-in in July 2023 (all zeros)."""
    data = _load(well_name="Wolfcamp A-1H")
    anomalies = data["Wolfcamp A-1H"]["anomalies"]
    shutins = [a for a in anomalies if a["type"] == "shut_in"]
    dates = [a["date"] for a in shutins]
    assert "2023-07-01" in dates


def test_shutin_detected_delaware():
    """Delaware Basin 3H has a shut-in in Nov 2022 (all zeros)."""
    data = _load(well_name="Delaware Basin 3H")
    anomalies = data["Delaware Basin 3H"]["anomalies"]
    shutins = [a for a in anomalies if a["type"] == "shut_in"]
    dates = [a["date"] for a in shutins]
    assert "2022-11-01" in dates


# --- Rate jump detection ---


def test_rate_jump_bone_spring():
    """Bone Spring 2H had a workover around April 2023 — rate jump should be detected."""
    data = _load(well_name="Bone Spring 2H")
    anomalies = data["Bone Spring 2H"]["anomalies"]
    jumps = [a for a in anomalies if a["type"] == "rate_jump"]
    # The jump should be around April 2023 (526.7 -> 670.3, ~27% increase)
    jump_dates = [a["date"] for a in jumps]
    assert "2023-04-01" in jump_dates
    # Verify magnitude is positive
    apr_jump = [a for a in jumps if a["date"] == "2023-04-01"][0]
    assert apr_jump["magnitude"] > 25


# --- Water cut calculation ---


def test_water_cut_calculation():
    """Verify water cut = water / (oil + water) * 100."""
    data = _load(well_name="Wolfcamp A-1H")
    monthly = data["Wolfcamp A-1H"]["monthly"]
    # First month: oil=1284.9, water=192.0
    # water_cut = 192.0 / (1284.9 + 192.0) * 100 = 12.998..
    expected_wc = round(192.0 / (1284.9 + 192.0) * 100, 2)
    assert monthly["water_cut"][0] == pytest.approx(expected_wc, abs=0.1)


def test_water_cut_zero_during_shutin():
    """Water cut should be None during a shut-in month."""
    data = _load(well_name="Wolfcamp A-1H")
    monthly = data["Wolfcamp A-1H"]["monthly"]
    # July 2023 is index 18 (month 19, 0-indexed 18)
    july_idx = monthly["dates"].index("2023-07-01")
    assert monthly["water_cut"][july_idx] is None


# --- GOR calculation ---


def test_gor_calculation():
    """Verify GOR = gas * 1000 / oil (scf/bbl)."""
    data = _load(well_name="Wolfcamp A-1H")
    monthly = data["Wolfcamp A-1H"]["monthly"]
    # First month: gas=2313, oil=1284.9
    expected_gor = round(2313 * 1000 / 1284.9, 2)
    assert monthly["gor"][0] == pytest.approx(expected_gor, abs=0.1)


def test_gor_none_during_shutin():
    """GOR should be None during a shut-in month."""
    data = _load(well_name="Wolfcamp A-1H")
    monthly = data["Wolfcamp A-1H"]["monthly"]
    july_idx = monthly["dates"].index("2023-07-01")
    assert monthly["gor"][july_idx] is None


# --- Summary and trends ---


def test_summary_fields():
    """Check that the summary contains required fields."""
    data = _load(well_name="Wolfcamp A-1H")
    summary = data["Wolfcamp A-1H"]["summary"]
    assert summary["num_months"] == 36
    assert summary["avg_oil_rate"] > 0
    assert summary["cumulative_oil"] > 0
    assert summary["current_water_cut"] is not None
    assert summary["current_gor"] is not None


def test_trend_directions():
    """Wolfcamp A-1H should show increasing water cut trend."""
    data = _load(well_name="Wolfcamp A-1H")
    trends = data["Wolfcamp A-1H"]["trends"]
    assert trends["water_cut_direction"] in ("increasing", "stable", "decreasing")
    assert trends["gor_direction"] in ("increasing", "stable", "decreasing")
    assert trends["avg_monthly_decline_pct"] is not None
    # Water cut should be increasing for this well (early ~13%, late ~50%)
    assert trends["water_cut_direction"] == "increasing"


# --- All wells ---


def test_all_wells_returned():
    """Without filtering, all three wells should be present."""
    data = _load()
    assert "Wolfcamp A-1H" in data
    assert "Bone Spring 2H" in data
    assert "Delaware Basin 3H" in data


# --- Edge cases ---


def test_nonexistent_well_returns_empty():
    """Filtering by a nonexistent well name returns an empty result."""
    data = _load(well_name="NoSuchWell")
    assert data == {}


def test_nonexistent_file_raises():
    """A nonexistent file path should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        analyze_production_trends("/nonexistent/path/data.csv")
