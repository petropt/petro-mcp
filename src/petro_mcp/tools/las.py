"""LAS file reading tools for the petro-mcp server."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lasio
import numpy as np

from petro_mcp.utils import validate_path


def _safe_value(v: Any) -> Any:
    """Convert a value to a JSON-serializable type."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.ndarray,)):
        return [_safe_value(x) for x in v]
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return v


def _read_lasio(path: Path) -> lasio.LASFile:
    """Open a LAS file with UTF-8 preference, falling back to latin-1.

    lasio defaults to latin-1, which silently produces mojibake for UTF-8
    files (common for non-ASCII well names).
    """
    try:
        return lasio.read(str(path), encoding="utf-8")
    except UnicodeDecodeError:
        return lasio.read(str(path), encoding="latin-1")


def _read_las(
    file_path: str,
    allowed_paths: Sequence[Path | str] | None = None,
) -> lasio.LASFile:
    """Read a LAS file with optional allowlist enforcement and encoding fallback."""
    path = Path(file_path)
    if path.suffix.lower() != ".las":
        raise ValueError(f"Not a LAS file: {file_path}")
    resolved = validate_path(path, allowed_paths)
    return _read_lasio(resolved)


def read_las_file(
    file_path: str,
    allowed_paths: Sequence[Path | str] | None = None,
) -> str:
    """Parse a LAS 2.0 file and return well header info and curve data summary.

    If the file is truncated or malformed, returns a degraded summary with
    ``status: "partial"`` instead of raising.

    Args:
        file_path: Path to the LAS file.
        allowed_paths: Optional allowlist of root directories. When provided,
            ``file_path`` must resolve inside one of them.

    Returns:
        JSON string with well header, curve summary, and gap statistics.
    """
    try:
        las = _read_las(file_path, allowed_paths)
    except (lasio.exceptions.LASDataError, IndexError, ValueError) as exc:
        # Not a parse failure for our purposes — return a degraded payload so
        # callers can keep walking through a directory of mixed-quality files.
        # Path/permission errors still propagate.
        if isinstance(exc, ValueError) and "Not a LAS file" in str(exc):
            raise
        return json.dumps({
            "file": file_path,
            "status": "partial",
            "warning": f"LAS file could not be fully parsed: {type(exc).__name__}: {exc}",
            "well_header": {},
            "num_curves": 0,
            "curves_summary": [],
            "depth_range": {"start": None, "stop": None, "step": None, "num_rows": 0},
        }, indent=2)

    header = {}
    for item in las.well:
        header[item.mnemonic] = {
            "value": _safe_value(item.value),
            "unit": item.unit,
            "descr": item.descr,
        }

    curves = []
    for curve in las.curves:
        data = curve.data
        num_points = int(len(data))
        if num_points > 0 and data.dtype.kind == "f":
            valid_mask = ~np.isnan(data)
            num_valid = int(valid_mask.sum())
        else:
            num_valid = num_points
        null_count = num_points - num_valid
        gap_pct = round(null_count / num_points * 100, 2) if num_points > 0 else 0.0
        curves.append({
            "mnemonic": curve.mnemonic,
            "unit": curve.unit,
            "descr": curve.descr,
            "num_points": num_points,
            "num_valid": num_valid,
            "null_count": null_count,
            "gap_pct": gap_pct,
            "min": _safe_value(np.nanmin(data)) if num_valid > 0 else None,
            "max": _safe_value(np.nanmax(data)) if num_valid > 0 else None,
            "mean": _safe_value(np.nanmean(data)) if num_valid > 0 else None,
        })

    result = {
        "file": file_path,
        "status": "ok",
        "version": las.version[0].value if las.version else "Unknown",
        "well_header": header,
        "num_curves": len(las.curves),
        "curves_summary": curves,
        "depth_range": {
            "start": _safe_value(las.index[0]) if len(las.index) > 0 else None,
            "stop": _safe_value(las.index[-1]) if len(las.index) > 0 else None,
            "step": _safe_value(las.well.STEP.value) if hasattr(las.well, "STEP") else None,
            "num_rows": len(las.index),
        },
    }
    return json.dumps(result, indent=2)


def get_curve_data(
    file_path: str,
    curve_names: list[str],
    start_depth: float | None = None,
    end_depth: float | None = None,
    max_samples: int = 0,
    allowed_paths: Sequence[Path | str] | None = None,
) -> str:
    """Get specific curve data from a LAS file with optional depth range.

    For curves that exceed ``max_samples`` points within the depth filter,
    the result is downsampled by taking every Nth sample (``N = ceil(total/
    max_samples)``). ``max_samples=0`` (default) disables the cap and returns
    every filtered point.

    Args:
        file_path: Path to the LAS file.
        curve_names: List of curve mnemonics to retrieve.
        start_depth: Optional start depth for filtering.
        end_depth: Optional end depth for filtering.
        max_samples: Cap on number of points per curve returned. Default 0
            (no cap). Set to a positive integer to enable downsampling.
        allowed_paths: Optional allowlist of root directories.

    Returns:
        JSON string with curve data arrays and sampling metadata.
    """
    las = _read_las(file_path, allowed_paths)

    available = {c.mnemonic for c in las.curves}
    missing = [c for c in curve_names if c not in available]
    if missing:
        raise ValueError(
            f"Curves not found in file: {missing}. Available: {sorted(available)}"
        )

    depth = las.index
    mask = np.ones(len(depth), dtype=bool)
    if start_depth is not None:
        mask &= depth >= start_depth
    if end_depth is not None:
        mask &= depth <= end_depth

    num_total = int(mask.sum())

    if max_samples > 0 and num_total > max_samples:
        sampling_factor = math.ceil(num_total / max_samples)
        # Build a downsample mask: keep every Nth index that's already in `mask`
        filtered_indices = np.where(mask)[0][::sampling_factor]
        out_mask = np.zeros(len(depth), dtype=bool)
        out_mask[filtered_indices] = True
    else:
        sampling_factor = 1
        out_mask = mask

    result: dict[str, Any] = {
        "depth": [_safe_value(d) for d in depth[out_mask]],
    }
    for name in curve_names:
        result[name] = [_safe_value(v) for v in las[name][out_mask]]

    num_returned = int(out_mask.sum())
    result["num_points_total"] = num_total
    result["num_points_returned"] = num_returned
    result["num_points"] = num_returned  # backward-compat with v1.0.0 callers
    result["sampling_factor"] = sampling_factor
    return json.dumps(result, indent=2)
