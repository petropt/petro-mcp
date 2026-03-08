"""Tests for production data tools."""

import json
import os

import pytest

from petro_mcp.tools.production import query_production_data

SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_production.csv")


def test_query_all_production():
    result = json.loads(query_production_data(SAMPLE_CSV))
    assert result["summary"]["num_records"] > 0
    assert len(result["summary"]["wells"]) == 3
    assert "Wolfcamp A-1H" in result["summary"]["wells"]
    assert "Bone Spring 2H" in result["summary"]["wells"]
    assert "Delaware Basin 3H" in result["summary"]["wells"]


def test_query_by_well():
    result = json.loads(query_production_data(SAMPLE_CSV, well_name="Wolfcamp A-1H"))
    assert result["summary"]["num_records"] == 36
    assert result["summary"]["wells"] == ["Wolfcamp A-1H"]


def test_query_by_date_range():
    result = json.loads(query_production_data(
        SAMPLE_CSV,
        well_name="Wolfcamp A-1H",
        start_date="2023-01-01",
        end_date="2023-06-30",
    ))
    assert result["summary"]["num_records"] == 6
    assert "oil_stats" in result["summary"]


def test_query_nonexistent_well():
    result = json.loads(query_production_data(SAMPLE_CSV, well_name="NoSuchWell"))
    assert result["summary"]["num_records"] == 0


def test_query_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        query_production_data("/nonexistent/file.csv")


def test_production_stats():
    result = json.loads(query_production_data(SAMPLE_CSV, well_name="Wolfcamp A-1H"))
    stats = result["summary"]["oil_stats"]
    assert stats["max"] > stats["min"]
    assert stats["avg"] > 0
    assert stats["total"] > 0
