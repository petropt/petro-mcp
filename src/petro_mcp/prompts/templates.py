"""Pre-built prompt templates for common petroleum engineering workflows."""

from __future__ import annotations

TEMPLATES: dict[str, dict[str, str]] = {
    "analyze_decline": {
        "name": "Analyze Decline Behavior",
        "description": "Analyze a well's production decline and estimate EUR",
        "template": (
            "Analyze the production decline behavior for this well.\n\n"
            "1. First, query the production data using query_production_data.\n"
            "2. Fit exponential, hyperbolic, and harmonic decline curves using fit_decline_curve.\n"
            "3. Compare the R-squared values to determine the best fit.\n"
            "4. Calculate EUR using the best-fit parameters with calculate_eur.\n"
            "5. Provide a summary including:\n"
            "   - Best-fit decline model and parameters (qi, Di, b-factor)\n"
            "   - Estimated Ultimate Recovery (EUR)\n"
            "   - Time to economic limit\n"
            "   - Current decline rate\n"
            "   - Any anomalies in the production data"
        ),
    },
    "compare_completions": {
        "name": "Compare Completions",
        "description": "Compare completion effectiveness across multiple wells",
        "template": (
            "Compare the completion effectiveness across the provided wells.\n\n"
            "For each well:\n"
            "1. Query production data using query_production_data.\n"
            "2. Fit decline curves to normalize for time on production.\n"
            "3. Calculate key metrics: IP30 (first 30-day avg), IP90, EUR.\n"
            "4. Calculate producing ratios (GOR, water cut) using calculate_ip_ratio.\n\n"
            "Then compare:\n"
            "- Initial productivity (qi) normalized by lateral length if available\n"
            "- Decline rates (Di, b-factor)\n"
            "- EUR per well\n"
            "- Water cut trends\n"
            "- Identify best and worst performers with possible explanations"
        ),
    },
    "summarize_logs": {
        "name": "Summarize Log Suite",
        "description": "Summarize the log curves in a LAS file",
        "template": (
            "Provide a comprehensive summary of this well's log suite.\n\n"
            "1. Read the LAS file using read_las_file to get an overview.\n"
            "2. List all available curves using list_curves.\n"
            "3. Get the well header using get_well_header.\n"
            "4. For key curves (GR, RHOB, NPHI, resistivity), get curve data and analyze:\n"
            "   - Depth intervals with high/low values\n"
            "   - Potential pay zones (low GR, high resistivity, crossover on density-neutron)\n"
            "   - Shale volume estimates from GR\n"
            "   - Any data quality issues (washouts, null values)\n"
            "5. Summarize the lithology interpretation and flag zones of interest"
        ),
    },
    "production_anomalies": {
        "name": "Identify Production Anomalies",
        "description": "Detect anomalies and changes in production patterns",
        "template": (
            "Analyze the production data for anomalies and operational events.\n\n"
            "1. Query all production data using query_production_data.\n"
            "2. Look for:\n"
            "   - Sudden rate changes (> 30% month-over-month)\n"
            "   - Extended shut-ins (zero production periods)\n"
            "   - GOR or water cut step changes (potential mechanical issues or water breakthrough)\n"
            "   - Production above decline trend (possible workovers or recompletions)\n"
            "   - Inverse decline (rate increasing over time)\n"
            "3. For each anomaly, note the date and magnitude.\n"
            "4. Suggest possible causes and recommended actions."
        ),
    },
    "calculate_well_eur": {
        "name": "Calculate EUR",
        "description": "Calculate Estimated Ultimate Recovery for a well",
        "template": (
            "Calculate the Estimated Ultimate Recovery (EUR) for this well.\n\n"
            "1. Query the production data using query_production_data.\n"
            "2. Fit a hyperbolic decline curve to get qi, Di, and b-factor.\n"
            "3. Calculate EUR using calculate_eur with an economic limit of 5 bbl/day.\n"
            "4. Report:\n"
            "   - Decline parameters with confidence intervals\n"
            "   - EUR in barrels\n"
            "   - Cumulative production to date vs EUR (recovery factor)\n"
            "   - Remaining reserves\n"
            "   - Time to economic limit"
        ),
    },
}


def get_template(name: str) -> dict[str, str] | None:
    """Get a prompt template by name."""
    return TEMPLATES.get(name)


def list_templates() -> list[dict[str, str]]:
    """List all available prompt templates."""
    return [
        {"name": key, "description": val["description"]}
        for key, val in TEMPLATES.items()
    ]
