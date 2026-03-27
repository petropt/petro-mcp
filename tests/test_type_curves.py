"""Tests for type curve generation and analysis tools."""

import csv
import json
import os
import tempfile

import numpy as np
import pytest

from petro_mcp.tools.type_curves import (
    calculate_eur_from_type_curve,
    calculate_normalized_production,
    calculate_vintage_curves,
    compare_well_to_type_curve,
    fit_type_curve,
    generate_type_curves,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic multi-well data
# ---------------------------------------------------------------------------

def _make_multi_well_csv(
    num_wells: int = 10,
    months: int = 36,
    qi_range: tuple[float, float] = (500, 1500),
    Di_range: tuple[float, float] = (0.03, 0.08),
    include_lateral_length: bool = False,
    include_vintage: bool = False,
    seed: int = 42,
) -> str:
    """Create a temp CSV with synthetic multi-well production data.

    Returns the file path.
    """
    rng = np.random.RandomState(seed)
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)

    fieldnames = ["date", "well_name", "oil", "gas", "water"]
    if include_lateral_length:
        fieldnames.append("lateral_length")
    if include_vintage:
        fieldnames.append("first_production_year")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for w in range(num_wells):
            qi = rng.uniform(*qi_range)
            Di = rng.uniform(*Di_range)
            b = rng.uniform(0.5, 1.5)
            ll = rng.uniform(5000, 10000) if include_lateral_length else None
            vintage = 2020 + (w % 3) if include_vintage else None
            start_year = 2020 + (w % 3)

            for m in range(months):
                rate = qi / (1 + b * Di * m) ** (1 / b)
                # Add some noise
                rate *= rng.uniform(0.9, 1.1)
                gas = rate * rng.uniform(1.5, 3.0)
                water = rate * rng.uniform(0.1, 0.5)
                row = {
                    "date": f"{start_year + m // 12}-{(m % 12) + 1:02d}-01",
                    "well_name": f"Well_{w + 1:02d}",
                    "oil": round(rate, 2),
                    "gas": round(gas, 2),
                    "water": round(water, 2),
                }
                if include_lateral_length:
                    row["lateral_length"] = round(ll, 0)
                if include_vintage:
                    row["first_production_year"] = vintage
                writer.writerow(row)
    return path


def _cleanup(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Tests: generate_type_curves
# ---------------------------------------------------------------------------

class TestGenerateTypeCurves:
    def test_basic_type_curves(self):
        path = _make_multi_well_csv()
        try:
            result = json.loads(generate_type_curves(path))
            assert "groups" in result
            assert "all_wells" in result["groups"]
            group = result["groups"]["all_wells"]
            assert "P10" in group
            assert "P50" in group
            assert "P90" in group
            assert group["num_wells"] == 10
            assert len(group["months"]) == 36
        finally:
            _cleanup(path)

    def test_custom_percentiles(self):
        path = _make_multi_well_csv()
        try:
            result = json.loads(generate_type_curves(path, percentiles=[25, 50, 75]))
            group = result["groups"]["all_wells"]
            assert "P25" in group
            assert "P50" in group
            assert "P75" in group
        finally:
            _cleanup(path)

    def test_p10_greater_than_p90(self):
        """P10 (optimistic) should have higher rates than P90 (conservative)."""
        path = _make_multi_well_csv(num_wells=20, seed=123)
        try:
            result = json.loads(generate_type_curves(path))
            group = result["groups"]["all_wells"]
            # Check first few non-None months
            for i in range(min(12, len(group["P10"]))):
                p10 = group["P10"][i]
                p90 = group["P90"][i]
                if p10 is not None and p90 is not None:
                    assert p10 >= p90, f"Month {i}: P10={p10} < P90={p90}"
        finally:
            _cleanup(path)

    def test_gas_rate_key(self):
        path = _make_multi_well_csv()
        try:
            result = json.loads(generate_type_curves(path, rate_key="gas"))
            assert result["rate_key"] == "gas"
            group = result["groups"]["all_wells"]
            assert "P50" in group
        finally:
            _cleanup(path)

    def test_empty_file_raises(self):
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        with open(path, "w") as f:
            f.write("date,well_name,oil\n")
        try:
            with pytest.raises(ValueError, match="No production records"):
                generate_type_curves(path)
        finally:
            _cleanup(path)

    def test_returns_json_string(self):
        path = _make_multi_well_csv()
        try:
            raw = generate_type_curves(path)
            assert isinstance(raw, str)
            json.loads(raw)  # Should not raise
        finally:
            _cleanup(path)

    def test_well_count_by_month(self):
        path = _make_multi_well_csv(num_wells=5)
        try:
            result = json.loads(generate_type_curves(path))
            group = result["groups"]["all_wells"]
            # All 5 wells should have data at month 0
            assert group["well_count_by_month"]["0"] == 5
        finally:
            _cleanup(path)


# ---------------------------------------------------------------------------
# Tests: calculate_normalized_production
# ---------------------------------------------------------------------------

class TestNormalizedProduction:
    def test_normalize_with_provided_lengths(self):
        path = _make_multi_well_csv(num_wells=3)
        try:
            lengths = {"Well_01": 5000, "Well_02": 10000, "Well_03": 7500}
            result = json.loads(calculate_normalized_production(
                path, lateral_lengths=lengths,
            ))
            assert result["num_wells"] == 3
            # Well with 5000 ft should have factor 0.2 (1000/5000)
            w1 = result["wells"]["Well_01"]
            assert abs(w1["normalization_factor"] - 0.2) < 0.001
        finally:
            _cleanup(path)

    def test_normalize_single_well(self):
        path = _make_multi_well_csv(num_wells=3)
        try:
            lengths = {"Well_01": 5000, "Well_02": 10000, "Well_03": 7500}
            result = json.loads(calculate_normalized_production(
                path, well_name="Well_02", lateral_lengths=lengths,
            ))
            assert result["num_wells"] == 1
            assert "Well_02" in result["wells"]
        finally:
            _cleanup(path)

    def test_no_lateral_lengths_raises(self):
        path = _make_multi_well_csv(num_wells=3)
        try:
            with pytest.raises(ValueError, match="No lateral length"):
                calculate_normalized_production(path)
        finally:
            _cleanup(path)

    def test_normalize_with_csv_column(self):
        path = _make_multi_well_csv(num_wells=3, include_lateral_length=True)
        try:
            result = json.loads(calculate_normalized_production(path))
            assert result["num_wells"] == 3
        finally:
            _cleanup(path)

    def test_nonexistent_well_raises(self):
        path = _make_multi_well_csv(num_wells=3)
        try:
            with pytest.raises(ValueError, match="No records found"):
                calculate_normalized_production(
                    path, well_name="Nonexistent",
                    lateral_lengths={"Nonexistent": 5000},
                )
        finally:
            _cleanup(path)


# ---------------------------------------------------------------------------
# Tests: calculate_vintage_curves
# ---------------------------------------------------------------------------

class TestVintageCurves:
    def test_vintage_from_column(self):
        path = _make_multi_well_csv(num_wells=9, include_vintage=True)
        try:
            result = json.loads(calculate_vintage_curves(path))
            assert "vintages" in result
            # Should have 3 vintages (2020, 2021, 2022)
            assert len(result["vintages"]) == 3
        finally:
            _cleanup(path)

    def test_vintage_inferred_from_date(self):
        path = _make_multi_well_csv(num_wells=6)
        try:
            result = json.loads(calculate_vintage_curves(path))
            assert "vintages" in result
            # Vintages are inferred from first production date
            assert len(result["vintages"]) >= 1
        finally:
            _cleanup(path)

    def test_vintage_well_counts(self):
        path = _make_multi_well_csv(num_wells=9, include_vintage=True)
        try:
            result = json.loads(calculate_vintage_curves(path))
            total_wells = sum(v["num_wells"] for v in result["vintages"].values())
            assert total_wells == 9
        finally:
            _cleanup(path)


# ---------------------------------------------------------------------------
# Tests: fit_type_curve
# ---------------------------------------------------------------------------

class TestFitTypeCurve:
    def test_fit_exponential(self):
        qi, Di = 1000.0, 0.05
        t = np.arange(36, dtype=float)
        rates = (qi * np.exp(-Di * t)).tolist()
        result = json.loads(fit_type_curve(rates, model="exponential"))
        assert result["model"] == "exponential"
        assert result["r_squared"] > 0.99
        assert abs(result["parameters"]["qi"] - qi) < 50

    def test_fit_hyperbolic(self):
        qi, Di, b = 1000.0, 0.05, 1.2
        t = np.arange(36, dtype=float)
        rates = (qi / (1 + b * Di * t) ** (1 / b)).tolist()
        result = json.loads(fit_type_curve(rates, model="hyperbolic"))
        assert result["model"] == "hyperbolic"
        assert result["r_squared"] > 0.99

    def test_fit_harmonic(self):
        qi, Di = 800.0, 0.04
        t = np.arange(36, dtype=float)
        rates = (qi / (1 + Di * t)).tolist()
        result = json.loads(fit_type_curve(rates, model="harmonic"))
        assert result["model"] == "harmonic"
        assert result["r_squared"] > 0.99

    def test_fit_with_explicit_times(self):
        qi, Di = 1000.0, 0.05
        times = [0, 3, 6, 9, 12, 18, 24, 30, 36]
        rates = [qi * np.exp(-Di * t) for t in times]
        result = json.loads(fit_type_curve(rates, percentile_times=times, model="exponential"))
        assert result["r_squared"] > 0.99

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            fit_type_curve([100, 90, 80], model="invalid")

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            fit_type_curve([100, 0])


# ---------------------------------------------------------------------------
# Tests: calculate_eur_from_type_curve
# ---------------------------------------------------------------------------

class TestEurFromTypeCurve:
    def test_single_case(self):
        params = {
            "P50": {"qi": 500.0, "Di": 0.05, "b": 1.0, "model": "hyperbolic"},
        }
        result = json.loads(calculate_eur_from_type_curve(params))
        assert "cases" in result
        assert "P50" in result["cases"]
        assert result["cases"]["P50"]["eur"] > 0

    def test_multiple_cases(self):
        params = {
            "P10": {"qi": 800.0, "Di": 0.04, "b": 1.2, "model": "hyperbolic"},
            "P50": {"qi": 500.0, "Di": 0.05, "b": 1.0, "model": "hyperbolic"},
            "P90": {"qi": 300.0, "Di": 0.07, "b": 0.8, "model": "hyperbolic"},
        }
        result = json.loads(calculate_eur_from_type_curve(params))
        p10_eur = result["cases"]["P10"]["eur"]
        p50_eur = result["cases"]["P50"]["eur"]
        p90_eur = result["cases"]["P90"]["eur"]
        assert p10_eur > p50_eur > p90_eur

    def test_exponential_model(self):
        params = {
            "P50": {"qi": 500.0, "Di": 0.05, "model": "exponential"},
        }
        result = json.loads(calculate_eur_from_type_curve(params))
        assert result["cases"]["P50"]["eur"] > 0

    def test_invalid_model_raises(self):
        params = {"P50": {"qi": 500.0, "Di": 0.05, "model": "bad_model"}}
        with pytest.raises(ValueError, match="Unknown model"):
            calculate_eur_from_type_curve(params)


# ---------------------------------------------------------------------------
# Tests: compare_well_to_type_curve
# ---------------------------------------------------------------------------

class TestCompareWellToTypeCurve:
    def _make_curves(self, n=24):
        t = np.arange(n, dtype=float)
        p10 = (1200 / (1 + 1.0 * 0.04 * t)).tolist()
        p50 = (800 / (1 + 1.0 * 0.05 * t)).tolist()
        p90 = (500 / (1 + 1.0 * 0.06 * t)).tolist()
        return p10, p50, p90

    def test_outperformer(self):
        p10, p50, p90 = self._make_curves()
        # Well above P10
        well = [p * 1.2 for p in p10]
        result = json.loads(compare_well_to_type_curve(well, p10, p50, p90))
        assert result["summary"]["overall_classification"] in ("outperformer", "above_average")
        assert result["summary"]["average_percentile"] < 30

    def test_underperformer(self):
        p10, p50, p90 = self._make_curves()
        # Well below P90
        well = [p * 0.5 for p in p90]
        result = json.loads(compare_well_to_type_curve(well, p10, p50, p90))
        assert result["summary"]["overall_classification"] in ("below_average", "underperformer")

    def test_average_well(self):
        p10, p50, p90 = self._make_curves()
        # Well right at P50
        result = json.loads(compare_well_to_type_curve(p50, p10, p50, p90))
        assert result["summary"]["average_percentile"] <= 55

    def test_monthly_detail_structure(self):
        p10, p50, p90 = self._make_curves(n=6)
        well = p50[:6]
        result = json.loads(compare_well_to_type_curve(well, p10, p50, p90))
        assert result["num_months"] == 6
        assert len(result["monthly_detail"]) == 6
        detail = result["monthly_detail"][0]
        assert "well_rate" in detail
        assert "p10" in detail
        assert "classification" in detail
        assert "percentile_estimate" in detail

    def test_cumulative_ratio(self):
        p10, p50, p90 = self._make_curves()
        # Well at P50 should have cum ratio near 1.0
        result = json.loads(compare_well_to_type_curve(p50, p10, p50, p90))
        ratio = result["summary"]["cumulative_vs_p50_ratio"]
        assert 0.95 <= ratio <= 1.05

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            compare_well_to_type_curve([], [], [], [])

    def test_mismatched_lengths_uses_minimum(self):
        p10, p50, p90 = self._make_curves(n=24)
        well = p50[:12]  # Only 12 months
        result = json.loads(compare_well_to_type_curve(well, p10, p50, p90))
        assert result["num_months"] == 12
