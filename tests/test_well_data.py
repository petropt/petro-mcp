"""Tests for well data resources."""
import json
from pathlib import Path

import pytest

from petro_mcp.resources.well_data import list_wells, production_summary

EXAMPLES_DIR = str(Path(__file__).parent.parent / "examples")
SAMPLE_CSV = str(Path(__file__).parent.parent / "examples" / "sample_production.csv")


class TestListWells:
    def test_list_wells_examples_dir(self):
        result = json.loads(list_wells(EXAMPLES_DIR))
        assert result["num_wells"] >= 1

    def test_list_wells_nonexistent_dir(self):
        with pytest.raises(FileNotFoundError):
            list_wells("/nonexistent/directory")

    def test_list_wells_empty_dir(self, tmp_path):
        result = json.loads(list_wells(str(tmp_path)))
        assert result["num_wells"] == 0


class TestProductionSummary:
    def test_production_summary(self):
        result = json.loads(production_summary(SAMPLE_CSV))
        assert result["num_wells"] == 3
        for well_name in result["summaries"]:
            summary = result["summaries"][well_name]
            assert summary["cumulative_oil"] > 0
            assert summary["num_records"] > 0

    def test_production_summary_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            production_summary("/nonexistent/file.csv")
