"""Production economics calculations for petroleum engineering.

Discounted cash flow analysis, NPV, IRR, PV10, breakeven price,
operating netback, payout period, and price sensitivity — all using
monthly time steps (the PE standard for production forecasts).

Pure stdlib math — no external dependencies required.
"""

from __future__ import annotations

import json
import math


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_positive(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _validate_fraction(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value}")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _monthly_npv(cash_flows: list[float], annual_rate: float) -> float:
    """Compute NPV from monthly cash flows at an annual discount rate.

    NPV = sum(CF_t / (1 + r/12)^t) for t = 0, 1, 2, ...
    """
    monthly_r = annual_rate / 12.0
    npv = 0.0
    for t, cf in enumerate(cash_flows):
        npv += cf / (1.0 + monthly_r) ** t
    return npv


def _bisect_irr(cash_flows: list[float], lo: float = -0.5, hi: float = 100.0,
                tol: float = 1e-8, max_iter: int = 200) -> float | None:
    """Find IRR by bisection — the annual rate where NPV = 0.

    Returns None if no sign change found in [lo, hi].
    """
    npv_lo = _monthly_npv(cash_flows, lo)
    npv_hi = _monthly_npv(cash_flows, hi)

    # Need a sign change
    if npv_lo * npv_hi > 0:
        return None

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        npv_mid = _monthly_npv(cash_flows, mid)
        if abs(npv_mid) < tol:
            return mid
        if npv_lo * npv_mid < 0:
            hi = mid
        else:
            lo = mid
            npv_lo = npv_mid
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# 1. Full well economics (DCF)
# ---------------------------------------------------------------------------

def calculate_well_economics(
    monthly_oil_bbl: list[float],
    monthly_gas_mcf: list[float],
    monthly_water_bbl: list[float],
    oil_price_bbl: float,
    gas_price_mcf: float,
    opex_monthly: float,
    capex: float,
    royalty_pct: float = 0.125,
    tax_rate: float = 0.0,
    discount_rate: float = 0.10,
    working_interest: float = 1.0,
    net_revenue_interest: float = 0.875,
) -> str:
    """Full discounted cash flow analysis for a well.

    Takes production arrays (from decline forecast) plus economic assumptions
    and returns NPV, IRR, payout period, and monthly cash flow details.

    Args:
        monthly_oil_bbl: Monthly oil production (bbl) for each period.
        monthly_gas_mcf: Monthly gas production (Mcf) for each period.
        monthly_water_bbl: Monthly water production (bbl) for each period.
        oil_price_bbl: Oil price ($/bbl).
        gas_price_mcf: Gas price ($/Mcf).
        opex_monthly: Monthly operating expense ($).
        capex: Total capital expenditure ($), applied at time 0.
        royalty_pct: Royalty fraction (0-1). Default 0.125 (12.5%).
        tax_rate: Severance/production tax rate (0-1). Default 0.0.
        discount_rate: Annual discount rate for NPV. Default 0.10 (10%).
        working_interest: Working interest fraction (0-1). Default 1.0.
        net_revenue_interest: Net revenue interest fraction (0-1). Default 0.875.

    Returns:
        JSON string with DCF results including NPV, IRR, payout_months,
        total_revenue, total_opex, total_net_cash_flow, monthly_cash_flow,
        cumulative_cash_flow, and profitability_index.
    """
    n = len(monthly_oil_bbl)
    if len(monthly_gas_mcf) != n or len(monthly_water_bbl) != n:
        raise ValueError(
            f"All production arrays must have the same length. "
            f"Got oil={n}, gas={len(monthly_gas_mcf)}, water={len(monthly_water_bbl)}"
        )
    if n == 0:
        raise ValueError("Production arrays must not be empty")

    _validate_non_negative("oil_price_bbl", oil_price_bbl)
    _validate_non_negative("gas_price_mcf", gas_price_mcf)
    _validate_non_negative("opex_monthly", opex_monthly)
    _validate_non_negative("capex", capex)
    _validate_fraction("royalty_pct", royalty_pct)
    _validate_fraction("tax_rate", tax_rate)
    _validate_non_negative("discount_rate", discount_rate)
    _validate_fraction("working_interest", working_interest)
    _validate_fraction("net_revenue_interest", net_revenue_interest)

    monthly_cf: list[float] = []
    cumulative_cf: list[float] = []
    total_revenue = 0.0
    total_opex_sum = 0.0
    cum = -capex * working_interest  # initial investment

    for t in range(n):
        gross_revenue = (
            monthly_oil_bbl[t] * oil_price_bbl
            + monthly_gas_mcf[t] * gas_price_mcf
        )
        net_revenue = gross_revenue * net_revenue_interest
        royalty = gross_revenue * royalty_pct
        tax = (net_revenue - royalty) * tax_rate if (net_revenue - royalty) > 0 else 0.0
        opex_wi = opex_monthly * working_interest

        net_cf = (net_revenue - royalty - tax - opex_wi) * working_interest
        # Apply WI to capex only at t=0 through the cumulative init above
        monthly_cf.append(round(net_cf, 2))
        total_revenue += gross_revenue
        total_opex_sum += opex_monthly

        cum += net_cf
        cumulative_cf.append(round(cum, 2))

    # Build full cash flow array with capex at t=0 for NPV/IRR
    full_cf = [-capex * working_interest] + monthly_cf

    npv = round(_monthly_npv(full_cf, discount_rate), 2)
    irr_val = _bisect_irr(full_cf)
    irr_pct = round(irr_val * 100, 2) if irr_val is not None else None

    # Payout: first month where cumulative >= 0
    payout = None
    for t, c in enumerate(cumulative_cf):
        if c >= 0:
            payout = t + 1  # 1-indexed month
            break

    # Profitability index
    capex_wi = capex * working_interest
    pi = round(npv / capex_wi + 1.0, 4) if capex_wi > 0 else None

    total_net = sum(monthly_cf)

    return json.dumps({
        "npv": npv,
        "irr_pct": irr_pct,
        "payout_months": payout,
        "profitability_index": pi,
        "total_gross_revenue": round(total_revenue, 2),
        "total_opex": round(total_opex_sum, 2),
        "total_net_cash_flow": round(total_net, 2),
        "capex": capex,
        "months": n,
        "monthly_cash_flow": monthly_cf,
        "cumulative_cash_flow": cumulative_cf,
        "inputs": {
            "oil_price_bbl": oil_price_bbl,
            "gas_price_mcf": gas_price_mcf,
            "opex_monthly": opex_monthly,
            "capex": capex,
            "royalty_pct": royalty_pct,
            "tax_rate": tax_rate,
            "discount_rate": discount_rate,
            "working_interest": working_interest,
            "net_revenue_interest": net_revenue_interest,
        },
        "method": "Discounted Cash Flow (monthly)",
        "units": {
            "npv": "USD",
            "irr": "percent annual",
            "cash_flow": "USD/month",
            "production": "bbl (oil/water), Mcf (gas)",
        },
    }, indent=2)


# ---------------------------------------------------------------------------
# 2. Simple NPV
# ---------------------------------------------------------------------------

def calculate_npv(
    cash_flows: list[float],
    discount_rate: float = 0.10,
) -> str:
    """Calculate Net Present Value from monthly cash flows.

    NPV = sum(CF_t / (1 + r/12)^t) for t = 0, 1, 2, ...

    Args:
        cash_flows: Monthly cash flows ($). First element is typically
            negative (capex). Subsequent elements are net monthly cash flows.
        discount_rate: Annual discount rate. Default 0.10 (10%).

    Returns:
        JSON string with NPV result.
    """
    if not cash_flows:
        raise ValueError("cash_flows must not be empty")
    _validate_non_negative("discount_rate", discount_rate)

    npv = round(_monthly_npv(cash_flows, discount_rate), 2)

    return json.dumps({
        "npv": npv,
        "periods": len(cash_flows),
        "inputs": {
            "discount_rate": discount_rate,
        },
        "method": "NPV = sum(CF_t / (1 + r/12)^t)",
        "units": {"npv": "USD", "discount_rate": "annual fraction"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 3. IRR
# ---------------------------------------------------------------------------

def calculate_irr(
    cash_flows: list[float],
) -> str:
    """Calculate Internal Rate of Return via bisection.

    IRR is the annual discount rate at which NPV = 0.

    Args:
        cash_flows: Monthly cash flows ($). First element is typically
            negative (capex).

    Returns:
        JSON string with IRR result (annual percentage).
    """
    if not cash_flows:
        raise ValueError("cash_flows must not be empty")

    irr_val = _bisect_irr(cash_flows)
    irr_pct = round(irr_val * 100, 2) if irr_val is not None else None

    return json.dumps({
        "irr_pct": irr_pct,
        "irr_annual": round(irr_val, 6) if irr_val is not None else None,
        "periods": len(cash_flows),
        "converged": irr_val is not None,
        "method": "Bisection solver (monthly cash flows)",
        "units": {"irr": "percent annual"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 4. PV10 (SEC standard)
# ---------------------------------------------------------------------------

def calculate_pv10(
    monthly_net_revenue: list[float],
) -> str:
    """Calculate PV10 — SEC standard present value at 10% annual discount.

    PV10 = sum(NR_t / (1.10)^(t/12))

    Args:
        monthly_net_revenue: Monthly net revenue ($) after royalties and opex.

    Returns:
        JSON string with PV10 result.
    """
    if not monthly_net_revenue:
        raise ValueError("monthly_net_revenue must not be empty")

    pv10 = 0.0
    for t, nr in enumerate(monthly_net_revenue):
        pv10 += nr / (1.10 ** (t / 12.0))

    return json.dumps({
        "pv10": round(pv10, 2),
        "total_undiscounted": round(sum(monthly_net_revenue), 2),
        "periods": len(monthly_net_revenue),
        "method": "SEC PV10: sum(NR_t / 1.10^(t/12))",
        "units": {"pv10": "USD"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 5. Breakeven price
# ---------------------------------------------------------------------------

def calculate_breakeven_price(
    monthly_production_bbl: list[float],
    monthly_opex: float,
    capex: float,
    discount_rate: float = 0.10,
    royalty_pct: float = 0.125,
    months: int | None = None,
) -> str:
    """Calculate breakeven oil price — the minimum price for NPV = 0.

    Uses bisection to find the oil price at which the discounted net cash
    flow equals zero.

    Args:
        monthly_production_bbl: Monthly oil production (bbl) per period.
        monthly_opex: Monthly operating expense ($).
        capex: Total capital expenditure ($).
        discount_rate: Annual discount rate. Default 0.10.
        royalty_pct: Royalty fraction (0-1). Default 0.125.
        months: Number of months to evaluate (default: len of production array).

    Returns:
        JSON string with breakeven_price_per_bbl.
    """
    if not monthly_production_bbl:
        raise ValueError("monthly_production_bbl must not be empty")
    _validate_non_negative("monthly_opex", monthly_opex)
    _validate_non_negative("capex", capex)
    _validate_non_negative("discount_rate", discount_rate)
    _validate_fraction("royalty_pct", royalty_pct)

    n = months if months is not None else len(monthly_production_bbl)
    prod = monthly_production_bbl[:n]

    def npv_at_price(price: float) -> float:
        cfs = [-capex]
        for t in range(len(prod)):
            revenue = prod[t] * price
            net = revenue * (1.0 - royalty_pct) - monthly_opex
            cfs.append(net)
        return _monthly_npv(cfs, discount_rate)

    # Bisection: find price where NPV = 0
    lo, hi = 0.0, 500.0

    # Expand upper bound if needed
    for _ in range(10):
        if npv_at_price(hi) > 0:
            break
        hi *= 2.0

    if npv_at_price(hi) <= 0:
        # Even at very high prices, NPV is negative (no production?)
        return json.dumps({
            "breakeven_price_per_bbl": None,
            "note": "No breakeven found — NPV remains negative even at high prices",
            "inputs": {
                "monthly_opex": monthly_opex,
                "capex": capex,
                "discount_rate": discount_rate,
                "royalty_pct": royalty_pct,
                "months": n,
            },
            "method": "Bisection solver",
            "units": {"breakeven_price": "USD/bbl"},
        }, indent=2)

    for _ in range(100):
        mid = (lo + hi) / 2.0
        if npv_at_price(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.01:
            break

    breakeven = round((lo + hi) / 2.0, 2)

    return json.dumps({
        "breakeven_price_per_bbl": breakeven,
        "npv_at_breakeven": round(npv_at_price(breakeven), 2),
        "inputs": {
            "monthly_opex": monthly_opex,
            "capex": capex,
            "discount_rate": discount_rate,
            "royalty_pct": royalty_pct,
            "months": n,
        },
        "method": "Bisection solver (oil price for NPV = 0)",
        "units": {"breakeven_price": "USD/bbl"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 6. Operating netback
# ---------------------------------------------------------------------------

def calculate_operating_netback(
    oil_price: float,
    gas_price: float,
    oil_rate_bpd: float,
    gas_rate_mcfd: float,
    opex_per_boe: float,
    royalty_pct: float = 0.125,
    transport_per_boe: float = 0.0,
) -> str:
    """Calculate operating netback per BOE.

    Netback = Revenue/BOE - Royalties/BOE - OPEX/BOE - Transport/BOE.
    Gas converted to BOE at 6 Mcf = 1 BOE.

    Args:
        oil_price: Oil price ($/bbl).
        gas_price: Gas price ($/Mcf).
        oil_rate_bpd: Oil production rate (bbl/day).
        gas_rate_mcfd: Gas production rate (Mcf/day).
        opex_per_boe: Operating expense per BOE ($/BOE).
        royalty_pct: Royalty fraction (0-1). Default 0.125.
        transport_per_boe: Transportation cost per BOE ($/BOE). Default 0.0.

    Returns:
        JSON string with netback breakdown per BOE.
    """
    _validate_non_negative("oil_price", oil_price)
    _validate_non_negative("gas_price", gas_price)
    _validate_non_negative("oil_rate_bpd", oil_rate_bpd)
    _validate_non_negative("gas_rate_mcfd", gas_rate_mcfd)
    _validate_non_negative("opex_per_boe", opex_per_boe)
    _validate_fraction("royalty_pct", royalty_pct)
    _validate_non_negative("transport_per_boe", transport_per_boe)

    # Convert gas to BOE (6 Mcf = 1 BOE)
    gas_boe = gas_rate_mcfd / 6.0
    total_boe = oil_rate_bpd + gas_boe

    if total_boe <= 0:
        return json.dumps({
            "netback_per_boe": 0.0,
            "note": "No production — zero BOE",
            "inputs": {
                "oil_price": oil_price,
                "gas_price": gas_price,
                "oil_rate_bpd": oil_rate_bpd,
                "gas_rate_mcfd": gas_rate_mcfd,
            },
            "method": "Operating netback",
            "units": {"netback": "USD/BOE"},
        }, indent=2)

    # Daily revenue
    daily_revenue = oil_rate_bpd * oil_price + gas_rate_mcfd * gas_price
    revenue_per_boe = daily_revenue / total_boe
    royalty_per_boe = revenue_per_boe * royalty_pct

    netback = revenue_per_boe - royalty_per_boe - opex_per_boe - transport_per_boe

    return json.dumps({
        "netback_per_boe": round(netback, 2),
        "revenue_per_boe": round(revenue_per_boe, 2),
        "royalty_per_boe": round(royalty_per_boe, 2),
        "opex_per_boe": round(opex_per_boe, 2),
        "transport_per_boe": round(transport_per_boe, 2),
        "total_boe_per_day": round(total_boe, 2),
        "oil_boe_per_day": round(oil_rate_bpd, 2),
        "gas_boe_per_day": round(gas_boe, 2),
        "inputs": {
            "oil_price": oil_price,
            "gas_price": gas_price,
            "oil_rate_bpd": oil_rate_bpd,
            "gas_rate_mcfd": gas_rate_mcfd,
            "opex_per_boe": opex_per_boe,
            "royalty_pct": royalty_pct,
            "transport_per_boe": transport_per_boe,
        },
        "method": "Operating netback (6 Mcf/BOE gas conversion)",
        "units": {"netback": "USD/BOE", "rates": "per day"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 7. Payout period
# ---------------------------------------------------------------------------

def calculate_payout_period(
    cash_flows: list[float],
) -> str:
    """Calculate payout period — months to recover initial investment.

    Payout is the first month where cumulative cash flow >= 0.

    Args:
        cash_flows: Monthly cash flows ($). First element is typically
            negative (capex).

    Returns:
        JSON string with payout_months.
    """
    if not cash_flows:
        raise ValueError("cash_flows must not be empty")

    cumulative = 0.0
    payout = None
    for t, cf in enumerate(cash_flows):
        cumulative += cf
        if cumulative >= 0:
            payout = t + 1  # 1-indexed
            break

    return json.dumps({
        "payout_months": payout,
        "total_cash_flow": round(sum(cash_flows), 2),
        "final_cumulative": round(sum(cash_flows), 2),
        "periods": len(cash_flows),
        "note": "Payout not reached within forecast period" if payout is None else None,
        "method": "First month with cumulative cash flow >= 0",
        "units": {"payout": "months", "cash_flow": "USD"},
    }, indent=2)


# ---------------------------------------------------------------------------
# 8. Price sensitivity
# ---------------------------------------------------------------------------

def calculate_price_sensitivity(
    monthly_oil_bbl: list[float],
    monthly_gas_mcf: list[float],
    monthly_water_bbl: list[float],
    opex_monthly: float,
    capex: float,
    price_scenarios: list[dict[str, float]],
    discount_rate: float = 0.10,
    royalty_pct: float = 0.125,
) -> str:
    """Calculate NPV across multiple price scenarios for sensitivity/tornado charts.

    Args:
        monthly_oil_bbl: Monthly oil production (bbl) per period.
        monthly_gas_mcf: Monthly gas production (Mcf) per period.
        monthly_water_bbl: Monthly water production (bbl) per period.
        opex_monthly: Monthly operating expense ($).
        capex: Total capital expenditure ($).
        price_scenarios: List of dicts, each with 'oil_price' and 'gas_price'.
        discount_rate: Annual discount rate. Default 0.10.
        royalty_pct: Royalty fraction (0-1). Default 0.125.

    Returns:
        JSON string with NPV at each price scenario.
    """
    n = len(monthly_oil_bbl)
    if len(monthly_gas_mcf) != n or len(monthly_water_bbl) != n:
        raise ValueError("All production arrays must have the same length")
    if n == 0:
        raise ValueError("Production arrays must not be empty")
    if not price_scenarios:
        raise ValueError("price_scenarios must not be empty")

    _validate_non_negative("opex_monthly", opex_monthly)
    _validate_non_negative("capex", capex)
    _validate_non_negative("discount_rate", discount_rate)
    _validate_fraction("royalty_pct", royalty_pct)

    results = []
    for scenario in price_scenarios:
        oil_p = scenario.get("oil_price", 0.0)
        gas_p = scenario.get("gas_price", 0.0)

        cfs = [-capex]
        for t in range(n):
            gross = monthly_oil_bbl[t] * oil_p + monthly_gas_mcf[t] * gas_p
            net = gross * (1.0 - royalty_pct) - opex_monthly
            cfs.append(net)

        npv = round(_monthly_npv(cfs, discount_rate), 2)
        results.append({
            "oil_price": oil_p,
            "gas_price": gas_p,
            "npv": npv,
        })

    return json.dumps({
        "scenarios": results,
        "num_scenarios": len(results),
        "inputs": {
            "opex_monthly": opex_monthly,
            "capex": capex,
            "discount_rate": discount_rate,
            "royalty_pct": royalty_pct,
            "months": n,
        },
        "method": "Price sensitivity (NPV at each scenario)",
        "units": {"npv": "USD", "prices": "USD/bbl and USD/Mcf"},
    }, indent=2)
