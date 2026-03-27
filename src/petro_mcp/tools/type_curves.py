"""Type curve generation and analysis tools for the petro-mcp server.

Generates P10/P50/P90 type curves from multi-well production data,
supports normalization by lateral length, vintage grouping, decline
curve fitting to percentile curves, and well-vs-type-curve comparison.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from petro_mcp.tools.production import _read_production_csv


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_csv_with_extras(file_path: str) -> list[dict[str, Any]]:
    """Read production CSV, preserving all extra columns beyond the standard set.

    _read_production_csv only maps date/well_name/oil/gas/water.  This wrapper
    reads the raw CSV and merges any additional columns (lateral_length,
    first_production_year, formation, etc.) into the parsed records.
    """
    import csv
    from pathlib import Path

    records = _read_production_csv(file_path)
    path = Path(file_path)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return records
        # Identify extra columns not already captured
        standard = {"date", "well_name", "oil", "gas", "water"}
        extra_fields = []
        for field in reader.fieldnames:
            fl = field.lower().strip()
            if fl not in standard and not any(
                fl in aliases for aliases in [
                    {"date", "production_date", "prod_date"},
                    {"well", "well_name", "wellname", "well name"},
                    {"oil", "oil_rate", "oil rate", "oil_bopd", "bopd"},
                    {"gas", "gas_rate", "gas rate", "gas_mcfd", "mcfd"},
                    {"water", "water_rate", "water rate", "water_bwpd", "bwpd"},
                ]
            ):
                extra_fields.append(field)

        if not extra_fields:
            return records

        # Re-read to get extra columns
        f.seek(0)
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < len(records):
                for field in extra_fields:
                    val = row.get(field, "").strip()
                    # Try to convert to float if numeric
                    try:
                        records[i][field.lower().strip()] = float(val) if val else None
                    except ValueError:
                        records[i][field.lower().strip()] = val if val else None

    return records

def _align_to_producing_month(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group records by well and assign producing month (0-indexed from first production).

    Returns a dict mapping well_name -> list of records with 'producing_month' added.
    """
    from petro_mcp.tools.production import _parse_date

    wells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        wn = r.get("well_name", "Unknown")
        wells[wn].append(r)

    result: dict[str, list[dict[str, Any]]] = {}
    for wn, recs in wells.items():
        # Sort by date
        try:
            recs.sort(key=lambda x: _parse_date(x["date"]))
        except (ValueError, KeyError):
            # If dates can't be parsed, assume already sorted
            pass
        for i, rec in enumerate(recs):
            rec["producing_month"] = i
        result[wn] = recs
    return result


def _compute_percentiles(
    well_data: dict[str, list[dict[str, Any]]],
    rate_key: str,
    percentiles: list[int | float],
) -> dict[str, Any]:
    """Compute percentile curves across wells at each producing month.

    Returns dict with 'months' and a key per percentile (e.g. 'P10', 'P50', 'P90').
    """
    # Find max producing month across all wells
    max_month = 0
    for recs in well_data.values():
        if recs:
            max_month = max(max_month, recs[-1]["producing_month"])

    months = list(range(max_month + 1))
    # Build rate matrix: each row is a well, each column is a producing month
    rate_matrix: dict[int, list[float]] = defaultdict(list)
    for recs in well_data.values():
        for rec in recs:
            val = rec.get(rate_key, 0.0)
            if val is None:
                val = 0.0
            rate_matrix[rec["producing_month"]].append(float(val))

    result_curves: dict[str, list[float | None]] = {f"P{int(p)}": [] for p in percentiles}
    valid_months: list[int] = []

    for mo in months:
        vals = rate_matrix.get(mo, [])
        if len(vals) < 2:
            # Need at least 2 wells at this month for meaningful percentiles
            for p in percentiles:
                result_curves[f"P{int(p)}"].append(None)
            continue
        valid_months.append(mo)
        for p in percentiles:
            # P10 = high performer (90th percentile of rates)
            # P90 = low performer (10th percentile of rates)
            # Oil industry convention: P10 is optimistic (high)
            pct_value = float(np.percentile(vals, 100 - p))
            result_curves[f"P{int(p)}"].append(round(pct_value, 2))

    return {
        "months": months,
        "well_count_by_month": {mo: len(rate_matrix.get(mo, [])) for mo in months},
        **result_curves,
    }


def _arps_hyperbolic(t: np.ndarray, qi: float, Di: float, b: float) -> np.ndarray:
    b = np.clip(b, 0.001, 2.0)
    return qi / (1 + b * Di * t) ** (1 / b)


def _arps_exponential(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
    return qi * np.exp(-Di * t)


def _arps_harmonic(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
    return qi / (1 + Di * t)


_TC_MODELS = {
    "hyperbolic": (_arps_hyperbolic, ["qi", "Di", "b"], [0, 0, 0], [np.inf, 10, 2.0]),
    "exponential": (_arps_exponential, ["qi", "Di"], [0, 0], [np.inf, 10]),
    "harmonic": (_arps_harmonic, ["qi", "Di"], [0, 0], [np.inf, 10]),
}


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------


def generate_type_curves(
    file_path: str,
    group_by: str | None = None,
    normalize_by: str | None = None,
    percentiles: list[int] | None = None,
    rate_key: str = "oil",
) -> str:
    """Generate P10/P50/P90 type curves from multi-well production data.

    Reads a multi-well CSV, aligns each well to time-zero (first production
    month), optionally groups by a column (vintage, formation, etc.), and
    computes percentile rate envelopes at each producing month.

    Args:
        file_path: Path to multi-well production CSV.
        group_by: Optional column name to group wells (e.g. 'formation',
            'vintage', 'lateral_length_bin'). If None, all wells in one group.
        normalize_by: Optional normalization column. If 'lateral_length',
            rates are normalized per 1000 ft.
        percentiles: Percentile levels to compute (default [10, 50, 90]).
            P10 = optimistic (high rate), P90 = conservative (low rate).
        rate_key: Which rate column to use ('oil', 'gas', or 'water').

    Returns:
        JSON string with type curve percentile arrays per group.
    """
    if percentiles is None:
        percentiles = [10, 50, 90]

    records = _read_csv_with_extras(file_path)
    if not records:
        raise ValueError("No production records found in file")

    # Check for grouping column
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if group_by:
        for r in records:
            gv = r.get(group_by, r.get(group_by.lower(), "Unknown"))
            if gv is None:
                gv = "Unknown"
            groups[str(gv)].append(r)
    else:
        groups["all_wells"] = records

    # Optional normalization
    lateral_lengths: dict[str, float] = {}
    if normalize_by == "lateral_length":
        # Try to extract lateral length from records (expects a lateral_length column)
        for r in records:
            wn = r.get("well_name", "Unknown")
            ll = r.get("lateral_length", r.get("lateral_length_ft"))
            if ll is not None and wn not in lateral_lengths:
                lateral_lengths[wn] = float(ll)

    result_groups: dict[str, Any] = {}
    for group_name, group_recs in groups.items():
        well_data = _align_to_producing_month(group_recs)

        # Normalize if requested
        if normalize_by == "lateral_length" and lateral_lengths:
            for wn, recs in well_data.items():
                ll = lateral_lengths.get(wn, 1000.0)
                factor = 1000.0 / ll if ll > 0 else 1.0
                for rec in recs:
                    if rate_key in rec:
                        rec[rate_key] = rec[rate_key] * factor

        curves = _compute_percentiles(well_data, rate_key, percentiles)
        curves["num_wells"] = len(well_data)
        result_groups[group_name] = curves

    result: dict[str, Any] = {
        "file": file_path,
        "rate_key": rate_key,
        "percentiles": percentiles,
        "group_by": group_by,
        "normalize_by": normalize_by,
        "groups": result_groups,
    }
    return json.dumps(result, indent=2)


def calculate_normalized_production(
    file_path: str,
    well_name: str | None = None,
    normalize_by: str = "lateral_length",
    lateral_lengths: dict[str, float] | None = None,
    rate_key: str = "oil",
    per_length: float = 1000.0,
) -> str:
    """Normalize production rates by lateral length for fair well comparison.

    Args:
        file_path: Path to production CSV.
        well_name: Optional single well to normalize. If None, all wells.
        normalize_by: Normalization basis (currently 'lateral_length').
        lateral_lengths: Dict mapping well_name -> lateral length in ft.
            If None, looks for 'lateral_length' column in CSV.
        rate_key: Rate column to normalize ('oil', 'gas', 'water').
        per_length: Normalize per this many feet (default 1000 = per 1000 ft).

    Returns:
        JSON string with normalized production records and summary.
    """
    records = _read_csv_with_extras(file_path)
    if not records:
        raise ValueError("No production records found in file")

    if well_name:
        records = [r for r in records if r.get("well_name", "").lower() == well_name.lower()]
        if not records:
            raise ValueError(f"No records found for well: {well_name}")

    # Build lateral length lookup
    ll_map: dict[str, float] = {}
    if lateral_lengths:
        ll_map = lateral_lengths
    else:
        for r in records:
            wn = r.get("well_name", "Unknown")
            ll = r.get("lateral_length", r.get("lateral_length_ft"))
            if ll is not None and wn not in ll_map:
                ll_map[wn] = float(ll)

    if not ll_map:
        raise ValueError(
            "No lateral length data available. Provide lateral_lengths dict "
            "or include 'lateral_length' column in CSV."
        )

    well_data = _align_to_producing_month(records)
    normalized_wells: dict[str, Any] = {}

    for wn, recs in well_data.items():
        ll = ll_map.get(wn)
        if ll is None or ll <= 0:
            continue
        factor = per_length / ll
        norm_rates = []
        for rec in recs:
            raw_rate = rec.get(rate_key, 0.0)
            norm_rate = round(raw_rate * factor, 2)
            norm_rates.append({
                "producing_month": rec["producing_month"],
                f"{rate_key}_raw": round(raw_rate, 2),
                f"{rate_key}_normalized": norm_rate,
            })
        normalized_wells[wn] = {
            "lateral_length_ft": ll,
            "normalization_factor": round(factor, 4),
            "records": norm_rates,
        }

    result = {
        "normalize_by": normalize_by,
        "per_length_ft": per_length,
        "rate_key": rate_key,
        "num_wells": len(normalized_wells),
        "wells": normalized_wells,
    }
    return json.dumps(result, indent=2)


def calculate_vintage_curves(
    file_path: str,
    vintage_column: str = "first_production_year",
    percentiles: list[int] | None = None,
    rate_key: str = "oil",
) -> str:
    """Generate type curves grouped by completion/vintage year.

    Tracks how well performance changes by vintage — are newer wells
    improving or declining relative to older completions?

    Args:
        file_path: Path to multi-well production CSV.
        vintage_column: Column name containing the vintage year
            (default 'first_production_year'). Alternatively, the vintage
            is inferred from each well's first production date.
        percentiles: Percentile levels (default [10, 50, 90]).
        rate_key: Rate column ('oil', 'gas', 'water').

    Returns:
        JSON string with type curves per vintage year.
    """
    if percentiles is None:
        percentiles = [10, 50, 90]

    records = _read_csv_with_extras(file_path)
    if not records:
        raise ValueError("No production records found in file")

    # Try to get vintage from column or infer from date
    from petro_mcp.tools.production import _parse_date

    well_vintage: dict[str, str] = {}
    has_vintage_col = any(vintage_column in r or vintage_column.lower() in r for r in records[:1])

    if has_vintage_col:
        for r in records:
            wn = r.get("well_name", "Unknown")
            if wn not in well_vintage:
                vv = r.get(vintage_column, r.get(vintage_column.lower()))
                if vv is not None:
                    well_vintage[wn] = str(vv)
    else:
        # Infer vintage from first production date per well
        well_first_date: dict[str, str] = {}
        for r in records:
            wn = r.get("well_name", "Unknown")
            if wn not in well_first_date:
                well_first_date[wn] = r["date"]
        for wn, date_str in well_first_date.items():
            try:
                dt = _parse_date(date_str)
                well_vintage[wn] = str(dt.year)
            except ValueError:
                well_vintage[wn] = "Unknown"

    # Group records by vintage
    vintage_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        wn = r.get("well_name", "Unknown")
        vintage = well_vintage.get(wn, "Unknown")
        vintage_groups[vintage].append(r)

    result_vintages: dict[str, Any] = {}
    for vintage, group_recs in sorted(vintage_groups.items()):
        well_data = _align_to_producing_month(group_recs)
        curves = _compute_percentiles(well_data, rate_key, percentiles)
        curves["num_wells"] = len(well_data)
        result_vintages[vintage] = curves

    result = {
        "file": file_path,
        "vintage_column": vintage_column,
        "rate_key": rate_key,
        "percentiles": percentiles,
        "vintages": result_vintages,
    }
    return json.dumps(result, indent=2)


def fit_type_curve(
    percentile_rates: list[float],
    percentile_times: list[float] | None = None,
    model: str = "hyperbolic",
) -> str:
    """Fit a decline model to a type curve percentile series.

    Takes a percentile rate array (e.g. P50 from generate_type_curves)
    and fits an Arps decline model to it.

    Args:
        percentile_rates: Rate values at each producing month.
        percentile_times: Time values (months). If None, assumed 0..N-1.
        model: Decline model ('exponential', 'hyperbolic', 'harmonic').

    Returns:
        JSON string with fitted parameters, R-squared, and predicted rates.
    """
    if model not in _TC_MODELS:
        raise ValueError(f"Unknown model: {model}. Must be one of: {list(_TC_MODELS.keys())}")

    rates = np.array(percentile_rates, dtype=float)

    if percentile_times is not None:
        times = np.array(percentile_times, dtype=float)
    else:
        times = np.arange(len(rates), dtype=float)

    # Remove None/NaN/zero values
    valid = np.isfinite(rates) & (rates > 0) & np.isfinite(times)
    rates = rates[valid]
    times = times[valid]

    if len(rates) < 3:
        raise ValueError("Need at least 3 valid rate points for curve fitting")

    func, param_names, lower, upper = _TC_MODELS[model]

    # Initial guesses scaled to data
    if model == "hyperbolic":
        p0 = [float(rates[0]), 0.05, 1.0]
    elif model == "exponential":
        p0 = [float(rates[0]), 0.05]
    else:  # harmonic
        p0 = [float(rates[0]), 0.05]

    try:
        popt, pcov = curve_fit(func, times, rates, p0=p0, bounds=(lower, upper), maxfev=10000)
    except RuntimeError as e:
        raise ValueError(f"Curve fitting failed: {e}") from e

    predicted = func(times, *popt)
    ss_res = np.sum((rates - predicted) ** 2)
    ss_tot = np.sum((rates - np.mean(rates)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    params = dict(zip(param_names, [round(float(p), 6) for p in popt]))

    result = {
        "model": model,
        "parameters": params,
        "r_squared": round(float(r_squared), 6),
        "num_points": len(rates),
        "predicted_rates": [round(float(v), 2) for v in predicted],
    }
    return json.dumps(result, indent=2)


def calculate_eur_from_type_curve(
    type_curve_params: dict[str, dict[str, float]],
    economic_limit: float = 5.0,
    max_time: float = 600,
) -> str:
    """Calculate EUR for each percentile case from type curve parameters.

    Args:
        type_curve_params: Dict mapping percentile label (e.g. 'P10', 'P50',
            'P90') to decline parameters dict with keys like 'qi', 'Di', 'b',
            and 'model'.
        economic_limit: Minimum economic rate (same units as qi).
        max_time: Maximum forecast months (default 600 = 50 years).

    Returns:
        JSON string with EUR for each percentile case.
    """
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    days_per_month = 30.44
    results: dict[str, Any] = {}

    for label, params in type_curve_params.items():
        model_name = params.get("model", "hyperbolic")
        if model_name not in _TC_MODELS:
            raise ValueError(f"Unknown model '{model_name}' for {label}")

        func, param_names, _, _ = _TC_MODELS[model_name]
        model_params = [params[p] for p in param_names]

        t = np.arange(0, max_time + 1, dtype=float)
        rates = func(t, *model_params)
        rates = np.maximum(rates, 0.0)

        # Find economic limit
        below = np.where(rates < economic_limit)[0]
        if len(below) > 0:
            econ_time = int(below[0])
            rates = rates[:econ_time + 1]
            t = t[:econ_time + 1]
        else:
            econ_time = int(max_time)

        eur = float(_trapz(rates, t)) * days_per_month

        results[label] = {
            "model": model_name,
            "parameters": {p: params[p] for p in param_names},
            "eur": round(eur, 0),
            "eur_unit": "bbl (or Mcf if gas)",
            "time_to_econ_limit_months": econ_time,
            "time_to_econ_limit_years": round(econ_time / 12, 1),
            "final_rate": round(float(rates[-1]), 2) if len(rates) > 0 else 0,
        }

    return json.dumps({
        "economic_limit": economic_limit,
        "max_time_months": max_time,
        "cases": results,
    }, indent=2)


def compare_well_to_type_curve(
    well_production: list[float],
    type_curve_p10: list[float],
    type_curve_p50: list[float],
    type_curve_p90: list[float],
) -> str:
    """Compare a single well's production to P10/P50/P90 type curves.

    Determines where the well sits relative to the type curve at each
    producing month — above P10 (outperformer), between P10-P50, between
    P50-P90, or below P90 (underperformer).

    Args:
        well_production: Well's monthly production rates.
        type_curve_p10: P10 (optimistic) type curve rates.
        type_curve_p50: P50 (median) type curve rates.
        type_curve_p90: P90 (conservative) type curve rates.

    Returns:
        JSON string with percentile ranking at each month and summary.
    """
    n = min(len(well_production), len(type_curve_p10), len(type_curve_p50), len(type_curve_p90))
    if n == 0:
        raise ValueError("All input arrays must have at least one element")

    well = np.array(well_production[:n], dtype=float)
    p10 = np.array(type_curve_p10[:n], dtype=float)
    p50 = np.array(type_curve_p50[:n], dtype=float)
    p90 = np.array(type_curve_p90[:n], dtype=float)

    classifications: list[str] = []
    percentile_estimates: list[float] = []

    for i in range(n):
        w, hi, mid, lo = well[i], p10[i], p50[i], p90[i]

        if w >= hi:
            classifications.append("above_P10")
            pct = 10.0  # Better than P10
        elif w >= mid:
            # Interpolate between P10 and P50
            if hi > mid and hi != mid:
                frac = (hi - w) / (hi - mid)
                pct = 10.0 + frac * 40.0
            else:
                pct = 30.0
            classifications.append("P10_to_P50")
        elif w >= lo:
            # Interpolate between P50 and P90
            if mid > lo and mid != lo:
                frac = (mid - w) / (mid - lo)
                pct = 50.0 + frac * 40.0
            else:
                pct = 70.0
            classifications.append("P50_to_P90")
        else:
            classifications.append("below_P90")
            pct = 90.0
        percentile_estimates.append(round(pct, 1))

    # Summary stats
    avg_pct = round(float(np.mean(percentile_estimates)), 1)
    recent_pct = round(float(np.mean(percentile_estimates[-min(6, n):])), 1)

    # Cumulative comparison
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    t = np.arange(n, dtype=float)
    cum_well = float(_trapz(well, t)) if n > 1 else float(well[0])
    cum_p50 = float(_trapz(p50, t)) if n > 1 else float(p50[0])
    cum_ratio = round(cum_well / cum_p50, 3) if cum_p50 > 0 else 0.0

    result = {
        "num_months": n,
        "monthly_detail": [
            {
                "month": i,
                "well_rate": round(float(well[i]), 2),
                "p10": round(float(p10[i]), 2),
                "p50": round(float(p50[i]), 2),
                "p90": round(float(p90[i]), 2),
                "classification": classifications[i],
                "percentile_estimate": percentile_estimates[i],
            }
            for i in range(n)
        ],
        "summary": {
            "average_percentile": avg_pct,
            "recent_6mo_percentile": recent_pct,
            "cumulative_vs_p50_ratio": cum_ratio,
            "overall_classification": (
                "outperformer" if avg_pct < 30
                else "above_average" if avg_pct < 50
                else "average" if avg_pct < 70
                else "below_average" if avg_pct < 90
                else "underperformer"
            ),
        },
    }
    return json.dumps(result, indent=2)
