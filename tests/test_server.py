"""Tests for server creation and tool group activation."""

from __future__ import annotations

import subprocess
import sys

import pytest

from petro_mcp.server import TOOL_GROUPS, create_server


def _tool_names(mcp) -> set[str]:
    """Extract registered tool names from a FastMCP instance."""
    # FastMCP stores tools in _tool_manager.tools dict
    return set(mcp._tool_manager._tools.keys())


class TestCreateServer:
    """Verify create_server() registers the correct tool groups."""

    def test_all_groups_by_default(self):
        """create_server() with no args loads every group."""
        server = create_server()
        names = _tool_names(server)
        # Spot-check a tool from each group
        assert "read_las" in names                  # las
        assert "query_production" in names           # production
        assert "fit_decline" in names                # decline
        assert "convert_oilfield_units" in names     # units
        assert "calculate_pvt_properties" in names   # pvt
        assert "calculate_vshale" in names           # petrophysics
        assert "pz_analysis" in names                # reservoir

    def test_single_group(self):
        """Only the requested group's tools are registered."""
        server = create_server({"las"})
        names = _tool_names(server)
        assert "read_las" in names
        assert "get_curve_values" in names
        # Other groups must be absent
        assert "fit_decline" not in names
        assert "query_production" not in names

    def test_multiple_groups(self):
        """Multiple groups can be loaded together."""
        server = create_server({"decline", "pvt"})
        names = _tool_names(server)
        assert "fit_decline" in names
        assert "calculate_eur" in names
        assert "calculate_pvt_properties" in names
        # las should be absent
        assert "read_las" not in names

    def test_empty_set_loads_nothing(self):
        """An explicit empty set loads no tools."""
        server = create_server(set())
        names = _tool_names(server)
        assert len(names) == 0

    def test_none_loads_all(self):
        """None explicitly means all groups."""
        server = create_server(None)
        all_server = create_server(set(TOOL_GROUPS.keys()))
        assert _tool_names(server) == _tool_names(all_server)


class TestListToolsCLI:
    """Test the --list-tools CLI flag."""

    def test_list_tools_output(self):
        result = subprocess.run(
            [sys.executable, "-m", "petro_mcp.server", "--list-tools"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        for group_name in TOOL_GROUPS:
            assert group_name in result.stdout
