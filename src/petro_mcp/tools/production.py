"""Production data tools for the petro-mcp server."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_date(date_str: str) -> datetime:
    """Parse a date string, trying common formats."""
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str}")


def _read_production_csv(file_path: str) -> list[dict[str, Any]]:
    """Read a production CSV file.

    Expected columns: date, well_name (optional), oil, gas, water.
    Column matching is case-insensitive and flexible.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Production file not found: {file_path}")

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV file: {file_path}")

        # Map columns flexibly
        col_map: dict[str, str] = {}
        for field in reader.fieldnames:
            fl = field.lower().strip()
            if "date" in fl:
                col_map["date"] = field
            elif fl in ("well", "well_name", "wellname", "well name"):
                col_map["well_name"] = field
            elif fl in ("oil", "oil_rate", "oil rate", "oil_bopd", "bopd"):
                col_map["oil"] = field
            elif fl in ("gas", "gas_rate", "gas rate", "gas_mcfd", "mcfd"):
                col_map["gas"] = field
            elif fl in ("water", "water_rate", "water rate", "water_bwpd", "bwpd"):
                col_map["water"] = field

        if "date" not in col_map:
            raise ValueError(f"No date column found in {file_path}. Columns: {reader.fieldnames}")

        records = []
        for row in reader:
            record: dict[str, Any] = {
                "date": row[col_map["date"]].strip(),
            }
            if "well_name" in col_map:
                record["well_name"] = row[col_map["well_name"]].strip()
            for key in ("oil", "gas", "water"):
                if key in col_map:
                    val = row[col_map[key]].strip()
                    record[key] = float(val) if val else 0.0
            records.append(record)
        return records


def query_production_data(
    file_path: str,
    well_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Query production data from a CSV file.

    Args:
        file_path: Path to the production CSV file.
        well_name: Optional well name to filter by.
        start_date: Optional start date (YYYY-MM-DD).
        end_date: Optional end date (YYYY-MM-DD).

    Returns:
        JSON string with filtered production records and summary statistics.
    """
    records = _read_production_csv(file_path)

    if well_name:
        records = [r for r in records if r.get("well_name", "").lower() == well_name.lower()]

    if start_date:
        sd = _parse_date(start_date)
        records = [r for r in records if _parse_date(r["date"]) >= sd]
    if end_date:
        ed = _parse_date(end_date)
        records = [r for r in records if _parse_date(r["date"]) <= ed]

    # Compute summary statistics
    oil_vals = [r.get("oil", 0) for r in records]
    gas_vals = [r.get("gas", 0) for r in records]
    water_vals = [r.get("water", 0) for r in records]

    summary = {
        "num_records": len(records),
        "wells": sorted(set(r.get("well_name", "Unknown") for r in records)),
    }
    if records:
        summary["date_range"] = {
            "first": records[0]["date"],
            "last": records[-1]["date"],
        }
        for name, vals in [("oil", oil_vals), ("gas", gas_vals), ("water", water_vals)]:
            if any(v > 0 for v in vals):
                summary[f"{name}_stats"] = {
                    "total": round(sum(vals), 2),
                    "avg": round(sum(vals) / len(vals), 2),
                    "max": round(max(vals), 2),
                    "min": round(min(vals), 2),
                    "last": round(vals[-1], 2),
                }

    result = {
        "summary": summary,
        "records": records,
    }

    from petro_mcp._pro import is_pro
    # Count unique wells in results
    wells = set()
    for r in records:
        wn = r.get("well_name") or r.get("well") or r.get("Well Name")
        if wn:
            wells.add(wn)

    if len(wells) > 5 and not is_pro():
        result["pro_hint"] = (
            f"You are analyzing {len(wells)} wells individually. "
            "PetroSuite Pro offers batch decline analysis, type curves, "
            "and Excel export for 100+ wells. See petropt.com/pro"
        )

    return json.dumps(result, indent=2)
