"""Multi-well LAS comparison."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lasio

from petro_mcp.tools.las import _read_lasio
from petro_mcp.utils import validate_path


def _well_attr(las: lasio.LASFile, key: str) -> str | None:
    """Look up a well-header attribute by mnemonic."""
    try:
        item = las.well.get(key) if hasattr(las.well, "get") else None
    except AttributeError:
        item = None
    if item is None:
        return None
    value = getattr(item, "value", None)
    if value in (None, ""):
        return None
    return str(value)


def _curve_units(las: lasio.LASFile) -> dict[str, str | None]:
    """Return {mnemonic: unit} for non-depth curves."""
    return {
        str(c.mnemonic): (str(c.unit) if c.unit else None)
        for c in las.curves
        if c.mnemonic != "DEPT"
    }


def _depth_range(las: lasio.LASFile) -> tuple[float | None, float | None]:
    depth = las.index
    if len(depth) == 0:
        return None, None
    return float(depth[0]), float(depth[-1])


def compare_well_logs(
    paths: Sequence[str],
    allowed_paths: Sequence[Path | str] | None = None,
) -> str:
    """Compare two or more LAS files: common curves, depth overlap, unit consistency.

    Args:
        paths: Two or more LAS file paths.
        allowed_paths: Optional allowlist of root directories.

    Returns:
        JSON string with per-well metadata, common curves, unit consistency
        per curve, depth overlap, and any flags.
    """
    if len(paths) < 2:
        raise ValueError("compare_well_logs requires at least 2 files")

    wells: list[dict[str, Any]] = []
    curve_units_per_well: list[dict[str, str | None]] = []
    depth_ranges: list[tuple[float | None, float | None]] = []

    for p in paths:
        # Path validation errors (PathNotAllowedError, FileNotFoundError)
        # propagate intentionally; parse failures degrade gracefully so a
        # single bad file does not abort the whole comparison.
        resolved = validate_path(p, allowed_paths)
        try:
            las = _read_lasio(resolved)
        except (lasio.exceptions.LASDataError, IndexError, KeyError, ValueError) as exc:
            wells.append({
                "file": str(p),
                "status": "unreadable",
                "warning": f"{type(exc).__name__}: {exc}",
                "well_name": None,
                "operator": None,
                "depth_start": None,
                "depth_stop": None,
                "curve_count": 0,
            })
            curve_units_per_well.append({})
            depth_ranges.append((None, None))
            continue
        dr = _depth_range(las)
        curve_map = _curve_units(las)
        wells.append({
            "file": str(p),
            "status": "ok",
            "well_name": _well_attr(las, "WELL"),
            "operator": _well_attr(las, "COMP"),
            "depth_start": dr[0],
            "depth_stop": dr[1],
            "curve_count": len(curve_map),
        })
        curve_units_per_well.append(curve_map)
        depth_ranges.append(dr)

    # Common curves = mnemonics present in EVERY well
    curve_sets = [set(m.keys()) for m in curve_units_per_well]
    common_curves = sorted(set.intersection(*curve_sets)) if curve_sets else []

    # Unit consistency per common curve
    unit_consistency: list[dict[str, Any]] = []
    for curve in common_curves:
        units_seen = sorted({
            (m[curve] or "") for m in curve_units_per_well
        })
        unit_consistency.append({
            "curve": curve,
            "units_seen": units_seen,
            "consistent": len(units_seen) == 1,
        })

    # Depth overlap across all wells (max-start, min-stop)
    starts = [s for s, _ in depth_ranges if s is not None]
    stops = [e for _, e in depth_ranges if e is not None]
    if starts and stops:
        overlap_start = max(starts)
        overlap_stop = min(stops)
        has_overlap = overlap_start <= overlap_stop
        depth_overlap = {
            "start": overlap_start,
            "stop": overlap_stop,
            "has_overlap": has_overlap,
        }
    else:
        depth_overlap = {"start": None, "stop": None, "has_overlap": False}

    # Flags
    flags: list[str] = []
    unreadable = [w["file"] for w in wells if w.get("status") == "unreadable"]
    if unreadable:
        flags.append(f"unreadable files: {', '.join(unreadable)}")
    if not common_curves:
        flags.append("no curves common to all files")
    mismatched = [u["curve"] for u in unit_consistency if not u["consistent"]]
    if mismatched:
        flags.append(f"unit mismatch on: {', '.join(mismatched)}")
    if not depth_overlap["has_overlap"]:
        flags.append("no overlapping depth interval across all files")

    return json.dumps({
        "num_files": len(paths),
        "wells": wells,
        "common_curves": common_curves,
        "unit_consistency": unit_consistency,
        "depth_overlap": depth_overlap,
        "flags": flags,
    }, indent=2)
