"""MCP resources for browsing well data."""

from __future__ import annotations

import json
from pathlib import Path

import lasio


def list_wells(directory: str) -> str:
    """List all wells found in a directory of LAS and CSV files.

    Args:
        directory: Path to directory containing well data files.

    Returns:
        JSON string with list of wells and their files.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    wells: dict[str, dict] = {}

    # Scan LAS files
    las_candidates = sorted(dir_path.glob("*.las")) + sorted(dir_path.glob("*.LAS"))
    las_seen: set[Path] = set()
    for las_file in las_candidates:
        resolved = las_file.resolve()
        if resolved in las_seen:
            continue
        las_seen.add(resolved)
        try:
            las = lasio.read(str(las_file))
            well_name = "Unknown"
            for item in las.well:
                if item.mnemonic in ("WELL", "WN"):
                    well_name = str(item.value).strip()
                    break
            if well_name not in wells:
                wells[well_name] = {"las_files": [], "csv_files": []}
            wells[well_name]["las_files"].append(str(las_file))
        except Exception:
            continue

    # Scan CSV files
    for csv_file in sorted(dir_path.glob("*.csv")):
        # Try to associate with a well by filename
        stem = csv_file.stem.replace("_", " ").replace("-", " ")
        matched = False
        for wn in wells:
            if wn.lower() in stem.lower():
                wells[wn]["csv_files"].append(str(csv_file))
                matched = True
                break
        if not matched:
            if stem not in wells:
                wells[stem] = {"las_files": [], "csv_files": []}
            wells[stem]["csv_files"].append(str(csv_file))

    result = {
        "directory": directory,
        "num_wells": len(wells),
        "wells": wells,
    }
    return json.dumps(result, indent=2)


def production_summary(file_path: str) -> str:
    """Get a production summary for wells in a CSV file.

    Args:
        file_path: Path to production CSV file.

    Returns:
        JSON string with per-well production summaries.
    """
    from petro_mcp.tools.production import _read_production_csv

    records = _read_production_csv(file_path)

    # Group by well
    by_well: dict[str, list] = {}
    for r in records:
        wn = r.get("well_name", "Unknown")
        by_well.setdefault(wn, []).append(r)

    summaries = {}
    for wn, recs in by_well.items():
        oil = [r.get("oil", 0) for r in recs]
        gas = [r.get("gas", 0) for r in recs]
        water = [r.get("water", 0) for r in recs]
        summaries[wn] = {
            "num_records": len(recs),
            "date_range": [recs[0]["date"], recs[-1]["date"]],
            "cumulative_oil": round(sum(oil) * 30.44, 0),
            "cumulative_gas": round(sum(gas) * 30.44, 0),
            "cumulative_water": round(sum(water) * 30.44, 0),
            "avg_oil_bopd": round(sum(oil) / len(oil), 1) if oil else 0,
            "avg_gas_mcfd": round(sum(gas) / len(gas), 1) if gas else 0,
            "last_oil_bopd": round(oil[-1], 1) if oil else 0,
            "last_gas_mcfd": round(gas[-1], 1) if gas else 0,
            "last_water_bwpd": round(water[-1], 1) if water else 0,
        }

    result = {
        "file": file_path,
        "num_wells": len(summaries),
        "summaries": summaries,
    }
    return json.dumps(result, indent=2)
