"""Tests for production economics tools."""

import json
import math

import pytest

from petro_mcp.tools.economics import (
    calculate_breakeven_price,
    calculate_irr,
    calculate_npv,
    calculate_operating_netback,
    calculate_payout_period,
    calculate_price_sensitivity,
    calculate_pv10,
    calculate_well_economics,
)


# ===========================================================================
# calculate_npv
# ===========================================================================

class TestCalculateNPV:
    """Tests for simple NPV calculation."""

    def test_known_npv_zero_rate(self):
        """At 0% discount, NPV = sum of cash flows."""
        cfs = [-1000.0, 200.0, 200.0, 200.0, 200.0, 200.0]
        result = json.loads(calculate_npv(cfs, discount_rate=0.0))
        assert result["npv"] == 0.0  # -1000 + 5*200 = 0

    def test_positive_npv(self):
        """Cash flows with clear positive NPV."""
        cfs = [-1000.0] + [100.0] * 24  # 24 months of $100
        result = json.loads(calculate_npv(cfs, discount_rate=0.10))
        # Undiscounted = -1000 + 2400 = 1400, discounted should be positive but less
        assert result["npv"] > 0
        assert result["npv"] < 1400.0

    def test_negative_npv(self):
        """Cash flows with negative NPV."""
        cfs = [-10000.0] + [100.0] * 12
        result = json.loads(calculate_npv(cfs, discount_rate=0.10))
        assert result["npv"] < 0

    def test_single_period(self):
        """Single cash flow — NPV equals that cash flow."""
        result = json.loads(calculate_npv([500.0], discount_rate=0.10))
        assert result["npv"] == 500.0

    def test_all_zero(self):
        """All-zero cash flows produce NPV = 0."""
        result = json.loads(calculate_npv([0.0, 0.0, 0.0], discount_rate=0.10))
        assert result["npv"] == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            calculate_npv([])

    def test_output_structure(self):
        result = json.loads(calculate_npv([-100.0, 50.0, 60.0]))
        assert "npv" in result
        assert "method" in result
        assert "units" in result
        assert "inputs" in result


# ===========================================================================
# calculate_irr
# ===========================================================================

class TestCalculateIRR:
    """Tests for IRR calculation via bisection."""

    def test_known_irr_simple(self):
        """Known case: -1000 then 12 months of ~$100 -> ~0% to moderate IRR."""
        # -1000 + 12*100 = 200 profit, IRR should be positive
        cfs = [-1000.0] + [100.0] * 12
        result = json.loads(calculate_irr(cfs))
        assert result["converged"] is True
        assert result["irr_pct"] is not None
        assert result["irr_pct"] > 0

    def test_irr_zero_npv_validation(self):
        """IRR should make NPV approximately zero."""
        cfs = [-5000.0] + [500.0] * 24
        result = json.loads(calculate_irr(cfs))
        irr = result["irr_annual"]
        assert irr is not None
        # Verify NPV at IRR is approximately 0
        from petro_mcp.tools.economics import _monthly_npv
        npv_at_irr = _monthly_npv(cfs, irr)
        assert abs(npv_at_irr) < 1.0  # within $1

    def test_no_sign_change(self):
        """All positive cash flows — no valid IRR."""
        cfs = [100.0, 200.0, 300.0]
        result = json.loads(calculate_irr(cfs))
        assert result["irr_pct"] is None
        assert result["converged"] is False

    def test_high_return(self):
        """High return over multiple months produces high IRR."""
        # -1000 then 12 months of $500 = large profit
        cfs = [-1000.0] + [500.0] * 12
        result = json.loads(calculate_irr(cfs))
        assert result["irr_pct"] is not None
        assert result["irr_pct"] > 50  # Should be high

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            calculate_irr([])


# ===========================================================================
# calculate_pv10
# ===========================================================================

class TestCalculatePV10:
    """Tests for SEC PV10 calculation."""

    def test_known_pv10(self):
        """Single period: PV10 at t=0 equals the value itself."""
        result = json.loads(calculate_pv10([1000.0]))
        assert result["pv10"] == 1000.0

    def test_discounting_reduces_value(self):
        """PV10 should be less than undiscounted total."""
        monthly = [1000.0] * 60  # 5 years
        result = json.loads(calculate_pv10(monthly))
        assert result["pv10"] < result["total_undiscounted"]
        assert result["pv10"] > 0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            calculate_pv10([])


# ===========================================================================
# calculate_breakeven_price
# ===========================================================================

class TestCalculateBreakevenPrice:
    """Tests for breakeven oil price calculation."""

    def test_known_breakeven(self):
        """With no capex and simple production, breakeven = opex / (prod * (1-royalty))."""
        prod = [1000.0] * 12  # 1000 bbl/month for 12 months
        opex = 5000.0  # $5000/month
        # At breakeven: price * 1000 * (1 - 0.125) = 5000
        # price = 5000 / 875 = $5.71
        result = json.loads(calculate_breakeven_price(prod, opex, capex=0.0,
                                                       discount_rate=0.0, royalty_pct=0.125))
        assert result["breakeven_price_per_bbl"] is not None
        assert abs(result["breakeven_price_per_bbl"] - 5.71) < 0.1

    def test_higher_capex_higher_breakeven(self):
        """Adding capex should increase breakeven price."""
        prod = [1000.0] * 24
        r1 = json.loads(calculate_breakeven_price(prod, 3000.0, capex=0.0))
        r2 = json.loads(calculate_breakeven_price(prod, 3000.0, capex=100000.0))
        assert r2["breakeven_price_per_bbl"] > r1["breakeven_price_per_bbl"]

    def test_zero_production(self):
        """Zero production means no breakeven possible."""
        prod = [0.0] * 12
        result = json.loads(calculate_breakeven_price(prod, 5000.0, capex=50000.0))
        assert result["breakeven_price_per_bbl"] is None

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            calculate_breakeven_price([], 1000.0, 50000.0)


# ===========================================================================
# calculate_operating_netback
# ===========================================================================

class TestCalculateOperatingNetback:
    """Tests for operating netback calculation."""

    def test_oil_only(self):
        """Oil-only well: netback = price * (1 - royalty) - opex."""
        result = json.loads(calculate_operating_netback(
            oil_price=70.0, gas_price=3.0,
            oil_rate_bpd=100.0, gas_rate_mcfd=0.0,
            opex_per_boe=15.0, royalty_pct=0.125,
        ))
        # Revenue/BOE = 70.0, royalty = 8.75, opex = 15
        expected = 70.0 - 8.75 - 15.0
        assert abs(result["netback_per_boe"] - expected) < 0.01

    def test_with_gas(self):
        """Oil + gas well: gas converted at 6 Mcf/BOE."""
        result = json.loads(calculate_operating_netback(
            oil_price=70.0, gas_price=3.0,
            oil_rate_bpd=100.0, gas_rate_mcfd=600.0,
            opex_per_boe=10.0, royalty_pct=0.125,
        ))
        # 100 bbl oil + 600/6 = 100 BOE gas = 200 BOE total
        assert result["total_boe_per_day"] == 200.0
        assert result["gas_boe_per_day"] == 100.0

    def test_zero_production(self):
        """Zero production returns zero netback."""
        result = json.loads(calculate_operating_netback(
            oil_price=70.0, gas_price=3.0,
            oil_rate_bpd=0.0, gas_rate_mcfd=0.0,
            opex_per_boe=15.0,
        ))
        assert result["netback_per_boe"] == 0.0

    def test_transport_cost(self):
        """Transport cost reduces netback."""
        r1 = json.loads(calculate_operating_netback(
            oil_price=70.0, gas_price=0.0,
            oil_rate_bpd=100.0, gas_rate_mcfd=0.0,
            opex_per_boe=10.0, transport_per_boe=0.0,
        ))
        r2 = json.loads(calculate_operating_netback(
            oil_price=70.0, gas_price=0.0,
            oil_rate_bpd=100.0, gas_rate_mcfd=0.0,
            opex_per_boe=10.0, transport_per_boe=5.0,
        ))
        assert r2["netback_per_boe"] == r1["netback_per_boe"] - 5.0


# ===========================================================================
# calculate_payout_period
# ===========================================================================

class TestCalculatePayoutPeriod:
    """Tests for payout period calculation."""

    def test_immediate_payout(self):
        """Positive first cash flow: payout in month 1."""
        result = json.loads(calculate_payout_period([100.0, 50.0]))
        assert result["payout_months"] == 1

    def test_typical_payout(self):
        """Typical: negative initial, then positive cash flows."""
        cfs = [-1000.0] + [200.0] * 12
        result = json.loads(calculate_payout_period(cfs))
        # Cumulative after 5 periods: -1000 + 1000 = 0
        assert result["payout_months"] == 6  # -1000, then 5*200=1000, cum=0 at t=6

    def test_never_payout(self):
        """Negative cash flows throughout — no payout."""
        cfs = [-1000.0] + [-50.0] * 12
        result = json.loads(calculate_payout_period(cfs))
        assert result["payout_months"] is None

    def test_single_negative(self):
        """Single negative cash flow — no payout."""
        result = json.loads(calculate_payout_period([-500.0]))
        assert result["payout_months"] is None

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            calculate_payout_period([])


# ===========================================================================
# calculate_well_economics
# ===========================================================================

class TestCalculateWellEconomics:
    """Tests for full DCF well economics."""

    def _simple_well(self, **overrides):
        """Helper to create a simple well economics scenario."""
        defaults = dict(
            monthly_oil_bbl=[3000.0] * 24,
            monthly_gas_mcf=[6000.0] * 24,
            monthly_water_bbl=[500.0] * 24,
            oil_price_bbl=70.0,
            gas_price_mcf=3.0,
            opex_monthly=15000.0,
            capex=500000.0,
            royalty_pct=0.125,
            tax_rate=0.0,
            discount_rate=0.10,
            working_interest=1.0,
            net_revenue_interest=0.875,
        )
        defaults.update(overrides)
        return json.loads(calculate_well_economics(**defaults))

    def test_output_structure(self):
        result = self._simple_well()
        assert "npv" in result
        assert "irr_pct" in result
        assert "payout_months" in result
        assert "profitability_index" in result
        assert "total_gross_revenue" in result
        assert "monthly_cash_flow" in result
        assert "cumulative_cash_flow" in result
        assert "method" in result
        assert "units" in result
        assert "inputs" in result

    def test_positive_economics(self):
        """Typical producing well should have positive NPV."""
        result = self._simple_well()
        assert result["npv"] > 0
        assert result["irr_pct"] is not None
        assert result["irr_pct"] > 0
        assert result["payout_months"] is not None

    def test_negative_economics(self):
        """Very high capex should produce negative NPV."""
        result = self._simple_well(capex=50_000_000.0)
        assert result["npv"] < 0

    def test_array_lengths(self):
        """Monthly arrays must be reported correctly."""
        result = self._simple_well()
        assert len(result["monthly_cash_flow"]) == 24
        assert len(result["cumulative_cash_flow"]) == 24

    def test_mismatched_arrays_raise(self):
        with pytest.raises(ValueError, match="same length"):
            calculate_well_economics(
                monthly_oil_bbl=[100.0] * 12,
                monthly_gas_mcf=[200.0] * 10,  # mismatch
                monthly_water_bbl=[50.0] * 12,
                oil_price_bbl=70.0,
                gas_price_mcf=3.0,
                opex_monthly=5000.0,
                capex=100000.0,
            )

    def test_empty_arrays_raise(self):
        with pytest.raises(ValueError, match="must not be empty"):
            calculate_well_economics(
                monthly_oil_bbl=[],
                monthly_gas_mcf=[],
                monthly_water_bbl=[],
                oil_price_bbl=70.0,
                gas_price_mcf=3.0,
                opex_monthly=5000.0,
                capex=100000.0,
            )

    def test_zero_capex(self):
        """With zero capex, PI should be None (avoid division by zero)."""
        result = self._simple_well(capex=0.0)
        assert result["profitability_index"] is None

    def test_higher_oil_price_higher_npv(self):
        r1 = self._simple_well(oil_price_bbl=50.0)
        r2 = self._simple_well(oil_price_bbl=80.0)
        assert r2["npv"] > r1["npv"]


# ===========================================================================
# calculate_price_sensitivity
# ===========================================================================

class TestCalculatePriceSensitivity:
    """Tests for price sensitivity analysis."""

    def test_multiple_scenarios(self):
        oil = [3000.0] * 12
        gas = [6000.0] * 12
        water = [500.0] * 12
        scenarios = [
            {"oil_price": 50.0, "gas_price": 2.0},
            {"oil_price": 70.0, "gas_price": 3.0},
            {"oil_price": 90.0, "gas_price": 4.0},
        ]
        result = json.loads(calculate_price_sensitivity(
            oil, gas, water, opex_monthly=10000.0, capex=200000.0,
            price_scenarios=scenarios,
        ))
        assert result["num_scenarios"] == 3
        # Higher prices should yield higher NPV
        npvs = [s["npv"] for s in result["scenarios"]]
        assert npvs[0] < npvs[1] < npvs[2]

    def test_empty_scenarios_raise(self):
        with pytest.raises(ValueError, match="must not be empty"):
            calculate_price_sensitivity(
                [100.0], [200.0], [50.0],
                opex_monthly=1000.0, capex=10000.0,
                price_scenarios=[],
            )

    def test_output_structure(self):
        result = json.loads(calculate_price_sensitivity(
            [1000.0], [2000.0], [100.0],
            opex_monthly=5000.0, capex=50000.0,
            price_scenarios=[{"oil_price": 60.0, "gas_price": 3.0}],
        ))
        assert "scenarios" in result
        assert "method" in result
        assert "units" in result
        assert "inputs" in result
        s = result["scenarios"][0]
        assert "oil_price" in s
        assert "gas_price" in s
        assert "npv" in s
