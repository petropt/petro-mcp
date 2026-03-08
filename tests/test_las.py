"""Tests for LAS file tools."""

import json
import os

import pytest

from petro_mcp.tools.las import get_curve_data, get_well_header, list_curves, read_las_file

SAMPLE_LAS = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_well.las")


def test_read_las_file():
    result = json.loads(read_las_file(SAMPLE_LAS))
    assert result["num_curves"] == 6
    assert result["depth_range"]["start"] == 5000.0
    assert result["depth_range"]["num_rows"] > 0
    assert "well_header" in result
    assert "curves_summary" in result


def test_get_well_header():
    result = json.loads(get_well_header(SAMPLE_LAS))
    assert "WELL" in result
    assert result["WELL"]["value"] == "WOLFCAMP A-1H"
    assert "UWI" in result
    assert "COMP" in result


def test_list_curves():
    result = json.loads(list_curves(SAMPLE_LAS))
    mnemonics = [c["mnemonic"] for c in result]
    assert "DEPT" in mnemonics
    assert "GR" in mnemonics
    assert "RHOB" in mnemonics
    assert "NPHI" in mnemonics
    assert "ILD" in mnemonics
    assert "SP" in mnemonics


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
