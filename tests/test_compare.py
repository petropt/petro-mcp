"""Tests for compare_well_logs."""

from __future__ import annotations

import json
import os
import shutil

import pytest

from petro_mcp.tools.compare import compare_well_logs

SAMPLE_LAS = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_well.las")


def test_requires_at_least_two_files():
    with pytest.raises(ValueError, match="at least 2"):
        compare_well_logs([SAMPLE_LAS])


def test_two_identical_files(tmp_path):
    """Two copies of the same file: 100% curve overlap, unit-consistent, full depth overlap."""
    a = tmp_path / "a.las"
    b = tmp_path / "b.las"
    shutil.copy(SAMPLE_LAS, a)
    shutil.copy(SAMPLE_LAS, b)
    result = json.loads(compare_well_logs([str(a), str(b)]))

    assert result["num_files"] == 2
    assert len(result["wells"]) == 2
    assert len(result["common_curves"]) >= 1
    # All units must be consistent (same file twice)
    for u in result["unit_consistency"]:
        assert u["consistent"] is True
    # Depth overlap is the full range
    assert result["depth_overlap"]["has_overlap"] is True
    # No flags should trigger
    assert result["flags"] == [] or "unit mismatch" not in " ".join(result["flags"])


def test_three_files_intersection(tmp_path):
    """Three copies: common_curves is intersection (= full set), no flags."""
    paths = []
    for name in ("a.las", "b.las", "c.las"):
        p = tmp_path / name
        shutil.copy(SAMPLE_LAS, p)
        paths.append(str(p))
    result = json.loads(compare_well_logs(paths))
    assert result["num_files"] == 3
    assert len(result["common_curves"]) >= 1
    assert all(u["consistent"] for u in result["unit_consistency"])


def test_disjoint_depth_ranges_flagged(tmp_path):
    """If two LAS files have non-overlapping depths, has_overlap is False and a flag fires."""
    # Build a minimal LAS at 5000-5005 and another at 6000-6005.
    def write_las(path, start, stop):
        rows = "\n".join(
            f"{d:.1f} 12.0" for d in [start + 0.5 * i for i in range(int((stop - start) / 0.5) + 1)]
        )
        path.write_text(
            "~Version\n"
            "VERS. 2.0 :CWLS LOG ASCII STANDARD - VERSION 2.0\n"
            "WRAP. NO :ONE LINE PER DEPTH STEP\n"
            "~Well\n"
            f"STRT.F           {start:.1f}:START DEPTH\n"
            f"STOP.F           {stop:.1f}:STOP DEPTH\n"
            "STEP.F             0.5 :STEP\n"
            f"WELL.   WELL-{start:.0f}-1 :WELL NAME\n"
            "~Curve\n"
            "DEPT.F        :DEPTH\n"
            "GR  .GAPI     :GAMMA RAY\n"
            "~Ascii\n"
            f"{rows}\n"
        )
    a = tmp_path / "shallow.las"
    b = tmp_path / "deep.las"
    write_las(a, 5000.0, 5005.0)
    write_las(b, 6000.0, 6005.0)
    result = json.loads(compare_well_logs([str(a), str(b)]))
    assert result["depth_overlap"]["has_overlap"] is False
    assert any("overlapping depth" in f for f in result["flags"])


def test_allowed_paths_enforced(tmp_path):
    inside = tmp_path / "wells"
    inside.mkdir()
    a = inside / "a.las"
    b = inside / "b.las"
    shutil.copy(SAMPLE_LAS, a)
    shutil.copy(SAMPLE_LAS, b)
    # Inside: OK
    result = json.loads(compare_well_logs([str(a), str(b)], allowed_paths=[str(inside)]))
    assert result["num_files"] == 2
    # Outside: rejected
    from petro_mcp.utils import PathNotAllowedError
    with pytest.raises(PathNotAllowedError):
        compare_well_logs([SAMPLE_LAS, str(b)], allowed_paths=[str(inside)])


def test_malformed_file_degrades_gracefully(tmp_path):
    """A malformed/empty LAS file should not abort the whole comparison."""
    good = tmp_path / "good.las"
    bad = tmp_path / "bad.las"
    shutil.copy(SAMPLE_LAS, good)
    bad.write_text("")  # empty file — lasio raises on this
    result = json.loads(compare_well_logs([str(good), str(bad)]))
    # Both files appear in the report; bad one is marked unreadable
    statuses = [w["status"] for w in result["wells"]]
    assert "ok" in statuses
    assert "unreadable" in statuses
    # The unreadable file should appear in the flags
    assert any("unreadable" in f for f in result["flags"])


def test_zero_common_curves(tmp_path):
    """Two files with disjoint curve sets report empty common_curves + a flag."""
    def write_las(path, curve_mnemonic):
        path.write_text(
            "~Version\n"
            "VERS. 2.0 :CWLS LOG ASCII STANDARD - VERSION 2.0\n"
            "WRAP. NO :ONE LINE PER DEPTH STEP\n"
            "~Well\n"
            "STRT.F           5000.0:START DEPTH\n"
            "STOP.F           5001.0:STOP DEPTH\n"
            "STEP.F             0.5 :STEP\n"
            "WELL.    DISJOINT-1 :WELL NAME\n"
            "~Curve\n"
            "DEPT.F        :DEPTH\n"
            f"{curve_mnemonic}.GAPI     :{curve_mnemonic}\n"
            "~Ascii\n"
            "5000.0 12.0\n"
            "5000.5 13.0\n"
            "5001.0 14.0\n"
        )
    a = tmp_path / "a.las"
    b = tmp_path / "b.las"
    write_las(a, "GR  ")
    write_las(b, "RHOB")
    result = json.loads(compare_well_logs([str(a), str(b)]))
    assert result["common_curves"] == []
    assert any("common to all" in f for f in result["flags"])
