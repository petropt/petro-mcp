"""Production trend analysis and anomaly detection tools."""

from __future__ import annotations

import json
from typing import Any

from petro_mcp.tools.production import _parse_date, _read_production_csv


def analyze_production_trends(
    file_path: str,
    well_name: str | None = None,
) -> str:
    """Analyze production trends and detect anomalies in production data.

    Computes water cut, GOR, oil decline rate, and cumulative production
    per well. Detects shut-ins, rate jumps/crashes, water breakthrough,
    and GOR blowouts.

    Args:
        file_path: Path to the production CSV file.
        well_name: Optional well name to filter by.

    Returns:
        JSON string with per-well summaries, trends, anomalies, and
        month-by-month water_cut / gor arrays.
    """
    records = _read_production_csv(file_path)

    # Group records by well
    wells: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        wn = rec.get("well_name", "Unknown")
        wells.setdefault(wn, []).append(rec)

    # Filter to a single well if requested
    if well_name is not None:
        filtered = {
            k: v for k, v in wells.items() if k.lower() == well_name.lower()
        }
        wells = filtered

    results: dict[str, Any] = {}

    for wn, recs in wells.items():
        # Sort by date
        recs.sort(key=lambda r: _parse_date(r["date"]))

        oil_vals = [r.get("oil", 0.0) for r in recs]
        gas_vals = [r.get("gas", 0.0) for r in recs]
        water_vals = [r.get("water", 0.0) for r in recs]
        dates = [r["date"] for r in recs]

        # --- Month-by-month metrics ---
        water_cuts: list[float | None] = []
        gors: list[float | None] = []
        cum_oil = 0.0
        cum_gas = 0.0
        cum_water = 0.0
        cum_oils: list[float] = []
        cum_gases: list[float] = []
        cum_waters: list[float] = []

        for i, (o, g, w) in enumerate(zip(oil_vals, gas_vals, water_vals)):
            # Water cut
            total_liquid = o + w
            if total_liquid > 0:
                wc = round(w / total_liquid * 100, 2)
            else:
                wc = None
            water_cuts.append(wc)

            # GOR (gas in Mcf, multiply by 1000 for scf/bbl)
            if o > 0:
                gor = round(g * 1000 / o, 2)
            else:
                gor = None
            gors.append(gor)

            # Cumulative
            cum_oil += o
            cum_gas += g
            cum_water += w
            cum_oils.append(round(cum_oil, 2))
            cum_gases.append(round(cum_gas, 2))
            cum_waters.append(round(cum_water, 2))

        # --- Oil decline rate (month-over-month % change) ---
        decline_rates: list[float | None] = [None]  # first month has no prior
        for i in range(1, len(oil_vals)):
            prev = oil_vals[i - 1]
            curr = oil_vals[i]
            if prev > 0:
                pct = round((curr - prev) / prev * 100, 2)
                decline_rates.append(pct)
            else:
                decline_rates.append(None)

        # --- Trend directions ---
        def _direction(series: list[float | None]) -> str:
            valid = [v for v in series if v is not None]
            if len(valid) < 2:
                return "insufficient_data"
            n = len(valid)
            first_half = valid[: n // 2]
            second_half = valid[n // 2 :]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            diff = avg_second - avg_first
            threshold = max(abs(avg_first) * 0.05, 1.0)
            if diff > threshold:
                return "increasing"
            elif diff < -threshold:
                return "decreasing"
            return "stable"

        wc_direction = _direction(water_cuts)
        gor_direction = _direction(gors)

        # Average monthly decline rate (only negative values = actual decline)
        valid_declines = [d for d in decline_rates if d is not None]
        avg_decline = (
            round(sum(valid_declines) / len(valid_declines), 2)
            if valid_declines
            else None
        )

        # --- Summary stats ---
        producing_oil = [v for v in oil_vals if v > 0]
        producing_gas = [v for v in gas_vals if v > 0]
        producing_water = [v for v in water_vals if v > 0]

        avg_oil = round(sum(producing_oil) / len(producing_oil), 2) if producing_oil else 0.0
        avg_gas = round(sum(producing_gas) / len(producing_gas), 2) if producing_gas else 0.0
        avg_water = round(sum(producing_water) / len(producing_water), 2) if producing_water else 0.0

        # Current (last month) water cut and GOR
        current_wc = water_cuts[-1] if water_cuts else None
        current_gor = gors[-1] if gors else None

        # --- Anomaly detection ---
        anomalies: list[dict[str, Any]] = []

        for i in range(len(recs)):
            o, g, w = oil_vals[i], gas_vals[i], water_vals[i]
            dt = dates[i]

            # Shut-in: all rates zero
            if o == 0 and g == 0 and w == 0:
                anomalies.append({
                    "type": "shut_in",
                    "date": dt,
                    "magnitude": 0,
                    "description": f"Well shut-in detected — zero production in {dt}",
                })

            if i > 0 and oil_vals[i - 1] > 0 and o > 0:
                pct_change = (o - oil_vals[i - 1]) / oil_vals[i - 1] * 100

                # Rate jump > 25% (captures workovers / restimulations)
                if pct_change > 25:
                    anomalies.append({
                        "type": "rate_jump",
                        "date": dt,
                        "magnitude": round(pct_change, 1),
                        "description": (
                            f"Oil rate increased {round(pct_change, 1)}% "
                            f"({round(oil_vals[i-1], 1)} -> {round(o, 1)} bopd) — "
                            f"possible workover or restimulation"
                        ),
                    })

                # Rate crash > 40%
                if pct_change < -40:
                    anomalies.append({
                        "type": "rate_crash",
                        "date": dt,
                        "magnitude": round(pct_change, 1),
                        "description": (
                            f"Oil rate decreased {round(abs(pct_change), 1)}% "
                            f"({round(oil_vals[i-1], 1)} -> {round(o, 1)} bopd) — "
                            f"possible mechanical failure"
                        ),
                    })

            # Water breakthrough: water cut jump > 10 pp
            if i > 0 and water_cuts[i] is not None and water_cuts[i - 1] is not None:
                wc_delta = water_cuts[i] - water_cuts[i - 1]
                if wc_delta > 10:
                    anomalies.append({
                        "type": "water_breakthrough",
                        "date": dt,
                        "magnitude": round(wc_delta, 1),
                        "description": (
                            f"Water cut jumped {round(wc_delta, 1)} percentage points "
                            f"({round(water_cuts[i-1], 1)}% -> {round(water_cuts[i], 1)}%)"
                        ),
                    })

            # GOR blowout: GOR increase > 50%
            if i > 0 and gors[i] is not None and gors[i - 1] is not None and gors[i - 1] > 0:
                gor_pct = (gors[i] - gors[i - 1]) / gors[i - 1] * 100
                if gor_pct > 50:
                    anomalies.append({
                        "type": "gor_blowout",
                        "date": dt,
                        "magnitude": round(gor_pct, 1),
                        "description": (
                            f"GOR increased {round(gor_pct, 1)}% "
                            f"({round(gors[i-1], 1)} -> {round(gors[i], 1)} scf/bbl)"
                        ),
                    })

        results[wn] = {
            "summary": {
                "num_months": len(recs),
                "avg_oil_rate": avg_oil,
                "avg_gas_rate": avg_gas,
                "avg_water_rate": avg_water,
                "cumulative_oil": cum_oils[-1] if cum_oils else 0,
                "cumulative_gas": cum_gases[-1] if cum_gases else 0,
                "cumulative_water": cum_waters[-1] if cum_waters else 0,
                "current_water_cut": current_wc,
                "current_gor": current_gor,
            },
            "trends": {
                "water_cut_direction": wc_direction,
                "gor_direction": gor_direction,
                "avg_monthly_decline_pct": avg_decline,
            },
            "anomalies": anomalies,
            "monthly": {
                "dates": dates,
                "water_cut": water_cuts,
                "gor": gors,
            },
        }

    from petro_mcp._pro import is_pro
    if results and not is_pro():
        results["web_tool"] = "Clean and analyze production data: https://tools.petropt.com/production-close/"

    return json.dumps(results, indent=2)
