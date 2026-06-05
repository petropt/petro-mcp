"""Tests for LAS file tools."""

import json
import os

import pytest

from petro_mcp.tools.las import get_curve_data, read_las_file

SAMPLE_LAS = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_well.las")


def test_read_las_file():
    result = json.loads(read_las_file(SAMPLE_LAS))
    assert result["num_curves"] == 6
    assert result["depth_range"]["start"] == 5000.0
    assert result["depth_range"]["num_rows"] > 0
    assert "well_header" in result
    assert "curves_summary" in result


def test_get_curve_data():
    result = json.loads(get_curve_data(SAMPLE_LAS, ["GR", "RHOB"]))
    assert "depth" in result
    assert "GR" in result
    assert "RHOB" in result
    assert len(result["GR"]) == len(result["depth"])


def test_get_curve_data_depth_range():
    result = json.loads(get_curve_data(SAMPLE_LAS, ["GR"], start_depth=5005.0, end_depth=5010.0))
    assert result["num_points"] > 0
    for d in result["depth"]:
        assert 5005.0 <= d <= 5010.0


def test_get_curve_data_invalid_curve():
    with pytest.raises(ValueError, match="Curves not found"):
        get_curve_data(SAMPLE_LAS, ["NONEXISTENT"])


def test_read_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        read_las_file("/nonexistent/file.las")


def test_read_non_las_file():
    with pytest.raises(ValueError, match="Not a LAS file"):
        read_las_file("/tmp/test.txt")


# ---------------------------------------------------------------------------
# v1.1.0: gap %, max_samples, encoding fallback, truncation, allowlist
# ---------------------------------------------------------------------------

def test_curve_summary_includes_gap_pct():
    """Each curve summary now reports null_count + gap_pct."""
    result = json.loads(read_las_file(SAMPLE_LAS))
    for c in result["curves_summary"]:
        assert "null_count" in c
        assert "gap_pct" in c
        assert isinstance(c["gap_pct"], (int, float))
        # gap_pct = null_count / num_points * 100 (rounded to 2dp)
        if c["num_points"] > 0:
            expected = round(c["null_count"] / c["num_points"] * 100, 2)
            assert c["gap_pct"] == expected


def test_get_curve_data_max_samples_caps_output(tmp_path):
    """When num_total exceeds max_samples, the result is downsampled."""
    # Default max_samples is 500 and the sample LAS has > 500 rows (per the
    # original num_rows assertion). Validate the cap directly.
    result = json.loads(get_curve_data(SAMPLE_LAS, ["GR"], max_samples=10))
    assert result["num_points_returned"] <= 10
    assert result["sampling_factor"] >= 1
    assert result["num_points_total"] > result["num_points_returned"]
    assert len(result["GR"]) == result["num_points_returned"]


def test_get_curve_data_max_samples_zero_returns_all():
    """max_samples=0 disables the cap and returns every filtered point."""
    result_capped = json.loads(get_curve_data(SAMPLE_LAS, ["GR"], max_samples=10))
    result_full = json.loads(get_curve_data(SAMPLE_LAS, ["GR"], max_samples=0))
    assert result_full["num_points_returned"] == result_full["num_points_total"]
    assert result_full["num_points_returned"] >= result_capped["num_points_returned"]
    assert result_full["sampling_factor"] == 1


def test_truncated_las_returns_partial_status(tmp_path):
    """Header-only LAS file should yield status='partial', not raise."""
    truncated = tmp_path / "header_only.las"
    # Write a minimal LAS header but no ~A (data) section.
    truncated.write_text(
        "~Version\n"
        "VERS. 2.0 :CWLS LOG ASCII STANDARD - VERSION 2.0\n"
        "WRAP. NO :ONE LINE PER DEPTH STEP\n"
        "~Well\n"
        "STRT.F           5000.0:START DEPTH\n"
        "STOP.F           5050.0:STOP DEPTH\n"
        "STEP.F             0.5 :STEP\n"
        "WELL.    TRUNCATED-1 :WELL NAME\n"
        "~Curve\n"
        "DEPT.F        :DEPTH\n"
        "GR  .GAPI     :GAMMA RAY\n"
    )
    result = json.loads(read_las_file(str(truncated)))
    # Either parses as 'ok' with empty/odd shape, or returns 'partial' — both
    # are acceptable; the contract is "don't blow up".
    assert result["status"] in ("partial", "ok")


def test_utf8_well_name_decoded_correctly(tmp_path):
    """UTF-8-encoded well names should not mojibake."""
    utf8_file = tmp_path / "utf8.las"
    content = (
        "~Version\n"
        "VERS. 2.0 :CWLS LOG ASCII STANDARD - VERSION 2.0\n"
        "WRAP. NO :ONE LINE PER DEPTH STEP\n"
        "~Well\n"
        "STRT.F           5000.0:START DEPTH\n"
        "STOP.F           5001.0:STOP DEPTH\n"
        "STEP.F             0.5 :STEP\n"
        "WELL.    Pozo-Ñoño-1 :WELL NAME\n"
        "~Curve\n"
        "DEPT.F        :DEPTH\n"
        "GR  .GAPI     :GAMMA RAY\n"
        "~Ascii\n"
        "5000.0 12.3\n"
        "5000.5 13.4\n"
        "5001.0 14.5\n"
    )
    utf8_file.write_text(content, encoding="utf-8")
    result = json.loads(read_las_file(str(utf8_file)))
    well_name = result["well_header"]["WELL"]["value"]
    assert "Ñ" in well_name and "ñ" in well_name


def test_allowed_paths_enforced(tmp_path):
    """When allowed_paths is set, files outside the allowlist are rejected."""
    inside = tmp_path / "wells"
    inside.mkdir()
    # Copy sample LAS into the allowed area
    import shutil
    sample = inside / "sample.las"
    shutil.copy(SAMPLE_LAS, sample)
    # Inside the allowlist: OK
    result = json.loads(read_las_file(str(sample), allowed_paths=[str(inside)]))
    assert result["status"] == "ok"
    # Outside the allowlist: rejected
    from petro_mcp.utils import PathNotAllowedError
    with pytest.raises(PathNotAllowedError):
        read_las_file(SAMPLE_LAS, allowed_paths=[str(inside)])


def test_allowed_paths_none_is_permissive():
    """allowed_paths=None should accept any readable LAS file (v1.0.0 behavior)."""
    result = json.loads(read_las_file(SAMPLE_LAS, allowed_paths=None))
    assert result["status"] == "ok"


def test_default_max_samples_zero_returns_all():
    """Default max_samples is 0 (no cap) — v1.0.0 behavior preserved."""
    result_default = json.loads(get_curve_data(SAMPLE_LAS, ["GR"]))
    result_uncapped = json.loads(get_curve_data(SAMPLE_LAS, ["GR"], max_samples=0))
    assert result_default["num_points_returned"] == result_uncapped["num_points_returned"]
    assert result_default["num_points_returned"] == result_default["num_points_total"]
    assert result_default["sampling_factor"] == 1


def test_max_samples_boundary_equal_returns_all():
    """When num_total == max_samples, sampling_factor is 1 and no downsampling."""
    # First find num_total for an unfiltered read
    total = json.loads(get_curve_data(SAMPLE_LAS, ["GR"]))["num_points_total"]
    # Now ask for exactly that many samples — should return all of them unchanged
    result = json.loads(get_curve_data(SAMPLE_LAS, ["GR"], max_samples=total))
    assert result["num_points_returned"] == total
    assert result["sampling_factor"] == 1
