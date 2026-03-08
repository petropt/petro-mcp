"""LAS file reading tools for the petro-mcp server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lasio
import numpy as np


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


def _read_las(file_path: str) -> lasio.LASFile:
    """Read a LAS file with error handling."""
    path = Path(file_path)
    if path.suffix.lower() not in (".las",):
        raise ValueError(f"Not a LAS file: {file_path}")
    if not path.exists():
        raise FileNotFoundError(f"LAS file not found: {file_path}")
    try:
        return lasio.read(str(path))
    except Exception as e:
        raise ValueError(f"Failed to parse LAS file: {e}") from e


def read_las_file(file_path: str) -> str:
    """Parse a LAS 2.0 file and return well header info and curve data summary.

    Args:
        file_path: Path to the LAS file.

    Returns:
        JSON string with well header and curve summary.
    """
    las = _read_las(file_path)

    header = {}
    for item in las.well:
        header[item.mnemonic] = {
            "value": _safe_value(item.value),
            "unit": item.unit,
            "descr": item.descr,
        }

    curves = []
    for curve in las.curves:
        data = curve.data[~np.isnan(curve.data)] if curve.data.dtype.kind == "f" else curve.data
        curves.append({
            "mnemonic": curve.mnemonic,
            "unit": curve.unit,
            "descr": curve.descr,
            "num_points": len(curve.data),
            "num_valid": len(data),
            "min": _safe_value(np.nanmin(curve.data)) if len(data) > 0 else None,
            "max": _safe_value(np.nanmax(curve.data)) if len(data) > 0 else None,
            "mean": _safe_value(np.nanmean(curve.data)) if len(data) > 0 else None,
        })

    result = {
        "file": file_path,
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


def get_well_header(file_path: str) -> str:
    """Extract well header metadata from a LAS file.

    Args:
        file_path: Path to the LAS file.

    Returns:
        JSON string with well header fields.
    """
    las = _read_las(file_path)
    header = {}
    for item in las.well:
        header[item.mnemonic] = {
            "value": _safe_value(item.value),
            "unit": item.unit,
            "descr": item.descr,
        }
    return json.dumps(header, indent=2)


def list_curves(file_path: str) -> str:
    """List all curves in a LAS file with units and descriptions.

    Args:
        file_path: Path to the LAS file.

    Returns:
        JSON string with list of curve info.
    """
    las = _read_las(file_path)
    curves = []
    for curve in las.curves:
        curves.append({
            "mnemonic": curve.mnemonic,
            "unit": curve.unit,
            "descr": curve.descr,
        })
    return json.dumps(curves, indent=2)


def get_curve_data(
    file_path: str,
    curve_names: list[str],
    start_depth: float | None = None,
    end_depth: float | None = None,
) -> str:
    """Get specific curve data from a LAS file with optional depth range.

    Args:
        file_path: Path to the LAS file.
        curve_names: List of curve mnemonics to retrieve.
        start_depth: Optional start depth for filtering.
        end_depth: Optional end depth for filtering.

    Returns:
        JSON string with curve data arrays.
    """
    las = _read_las(file_path)

    available = {c.mnemonic for c in las.curves}
    missing = [c for c in curve_names if c not in available]
    if missing:
        raise ValueError(f"Curves not found in file: {missing}. Available: {sorted(available)}")

    depth = las.index
    mask = np.ones(len(depth), dtype=bool)
    if start_depth is not None:
        mask &= depth >= start_depth
    if end_depth is not None:
        mask &= depth <= end_depth

    result: dict[str, Any] = {
        "depth": [_safe_value(d) for d in depth[mask]],
    }
    for name in curve_names:
        data = las[name][mask]
        result[name] = [_safe_value(v) for v in data]

    result["num_points"] = int(mask.sum())
    return json.dumps(result, indent=2)
