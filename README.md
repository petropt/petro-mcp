# petro-mcp

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple.svg)](https://modelcontextprotocol.io)

**MCP server that gives LLMs access to petroleum engineering data and tools.**

Parse well logs, query production data, fit decline curves, calculate EUR, and run nodal analysis -- all through natural language with any MCP-compatible AI assistant.

Built by [Groundwork Analytics](https://petropt.com)

---

## What is this?

petro-mcp is a [Model Context Protocol](https://modelcontextprotocol.io) server that exposes petroleum engineering workflows to LLMs like Claude and other MCP-compatible assistants. Instead of writing scripts to parse LAS files or fit decline curves, just ask your AI assistant in plain English.

**Why MCP?** MCP is an open standard that lets AI assistants interact with external data and tools. While other energy MCP servers provide commodity prices ([OilpriceAPI](https://github.com/OilpriceAPI/mcp-server)), petro-mcp is purpose-built for petroleum engineering workflows -- LAS well log parsing, physics-constrained decline curve analysis, nodal analysis, and production diagnostics.

## Prerequisites

- **Python 3.10 or later** ([download](https://www.python.org/downloads/))
- **Claude Desktop** ([download](https://claude.ai/download)), **Cursor** ([download](https://cursor.sh/)), or any MCP-compatible client

## Installation

```bash
pip install petro-mcp
```

Or install from source:

```bash
git clone https://github.com/petropt/petro-mcp.git
cd petro-mcp
pip install -e .
```

## Quick Start

### Configure with Claude Desktop

Add to your `claude_desktop_config.json`:

> **Config file location:**
> - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
> - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "petro-mcp": {
      "command": "petro-mcp",
      "args": []
    }
  }
}
```

If installed from source:

```json
{
  "mcpServers": {
    "petro-mcp": {
      "command": "python",
      "args": ["-m", "petro_mcp.server"]
    }
  }
}
```

### Configure with Cursor / VS Code

Add to your MCP settings (`.cursor/mcp.json` or equivalent):

```json
{
  "mcpServers": {
    "petro-mcp": {
      "command": "petro-mcp"
    }
  }
}
```

### Use it

Once configured, ask your AI assistant:

- *"Read the LAS file at /data/wells/wolfcamp.las and summarize the curves"*
- *"Load production data from /data/prod.csv and fit a decline curve for Wolfcamp A-1H"*
- *"Calculate EUR with qi=800, Di=0.06, b=1.1 and a 5 bbl/day economic limit"*
- *"Run a nodal analysis: 3500 psi reservoir, PI=4.2, 2.875-inch tubing, 150 psi wellhead"*

## Tools

| Tool | Description |
|------|-------------|
| `read_las` | Parse a LAS 2.0 file -- returns well header and curve statistics |
| `get_header` | Extract well header metadata (name, UWI, location, KB, TD) |
| `get_curves` | List all curves with units and descriptions |
| `get_curve_values` | Get curve data with optional depth range filtering |
| `query_production` | Query production CSV data by well name and date range |
| `fit_decline` | Fit Arps decline curves (exponential, hyperbolic, harmonic) |
| `calculate_eur` | Calculate Estimated Ultimate Recovery from decline parameters |
| `calculate_ratios` | Compute GOR, WOR, water cut; classify well type |
| `run_nodal_analysis` | Simplified IPR/VLP intersection for operating point |
| `analyze_trends` | Detect production anomalies: shut-ins, rate jumps, water breakthrough |
| `calculate_pvt_properties` | Black-oil PVT: Pb, Bo, Rs, viscosity, Z-factor (Standing, Beggs-Robinson) |
| `calculate_bubble_point` | Bubble point pressure from Standing's correlation |
| `convert_oilfield_units` | Convert between oilfield units (pressure, volume, density, API/SG, BOE) |
| `list_oilfield_units` | List all supported unit categories and conversions |

### Decline Curve Analysis

The decline curve tools are physics-constrained:

- **b-factor** bounded to [0, 2] (physical range for Arps models)
- **Non-negative rates** enforced throughout
- **Three models**: exponential (b=0), harmonic (b=1), hyperbolic (0 < b < 2)
- Returns R-squared, parameter uncertainties, and residual statistics

### Production Data

Accepts CSV files with flexible column naming:
- Date column: `date`, `Date`, `DATE`
- Oil: `oil`, `oil_rate`, `BOPD`
- Gas: `gas`, `gas_rate`, `MCFD`
- Water: `water`, `water_rate`, `BWPD`
- Well name: `well_name`, `well`, `Well Name`

## Prompt Templates

Built-in prompt templates for common workflows:

| Prompt | Description |
|--------|-------------|
| `analyze_decline` | Full decline analysis with model comparison and EUR |
| `compare_completions` | Compare completion effectiveness across wells |
| `summarize_logs` | Summarize log suite with pay zone identification |
| `production_anomalies` | Detect rate changes, shut-ins, water breakthrough |
| `calculate_well_eur` | Step-by-step EUR calculation with confidence intervals |

## Resources

| Resource URI | Description |
|-------------|-------------|
| `wells://list/{directory}` | Browse wells in a directory of LAS/CSV files |
| `wells://production/{file_path}` | Per-well production summary from a CSV file |

## Example Data

The `examples/` directory includes:
- `sample_well.las` -- Synthetic Wolfcamp well log (GR, RHOB, NPHI, ILD, SP)
- `sample_production.csv` -- 36 months of decline data for 3 Permian Basin wells
- `usage_examples.md` -- Detailed usage examples

## Example Output

> **Prompt:** *"Read the LAS file at examples/sample_well.las and summarize the curves"*

```
Well: Wolfcamp A-1H (Spraberry Trend, Midland County, TX)
Depth range: 5000–5020 ft, 0.5 ft step (41 samples)

┌───────┬──────┬────────────────────────────┬───────────────┐
│ Curve │ Unit │        Description         │     Range     │
├───────┼──────┼────────────────────────────┼───────────────┤
│ GR    │ GAPI │ Gamma Ray                  │ 22–126        │
│ RHOB  │ G/CC │ Bulk Density               │ 2.53–2.67     │
│ NPHI  │ V/V  │ Neutron Porosity           │ 0.09–0.23     │
│ ILD   │ OHMM │ Deep Induction Resistivity │ 3.9–175       │
│ SP    │ MV   │ Spontaneous Potential      │ -30.5 to -2.0 │
└───────┴──────┴────────────────────────────┴───────────────┘

Quick interpretation:
- Shale zones (~5002–5005 ft): High GR (>100), low resistivity,
  high NPHI — classic shale signature.
- Clean/tight zones (~5008–5010 ft): Low GR (<30), high resistivity
  (>100 ohm-m), low NPHI — likely tight carbonate, potentially
  hydrocarbon-bearing.
```

## Limitations

- **LAS 2.0 only** -- LAS 3.0 and DLIS formats are not yet supported
- **Arps models only** -- Modified hyperbolic, Duong, and SEPD decline models are not yet implemented
- **Simplified nodal analysis** -- Uses Vogel IPR and a simplified friction model, not Beggs-Brill or Hagedorn-Brown. Results are directional/screening-level, not suitable for detailed well design
- **CSV production data only** -- No database, WITSML, or API connectors yet
- **No visualization** -- Returns JSON data only, no plots

## Development

```bash
git clone https://github.com/petropt/petro-mcp.git
cd petro-mcp
pip install -e .
pip install pytest
pytest tests/ -v
```

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

Areas where contributions are especially valuable:
- WITSML real-time data support
- Reservoir simulation output readers (Eclipse, OPM Flow)
- OSDU API connector
- Additional decline curve models (modified hyperbolic, Duong, SEPD)
- Well spacing and interference analysis

## License

MIT License. See [LICENSE](LICENSE) for details.

## Need More?

petro-mcp covers single-well analysis workflows. For advanced capabilities, Groundwork Analytics offers:

- Multi-well batch analysis and ranking
- Probabilistic EUR (P10/P50/P90)
- Database integration (Enverus, IHS, WITSML)
- Custom AI workflows for your engineering team

Visit [petropt.com](https://petropt.com) or reach out at info@petropt.com

## About

Built by [Groundwork Analytics](https://petropt.com)

- Website: [petropt.com](https://petropt.com)
- LinkedIn: [Groundwork Analytics](https://www.linkedin.com/company/groundworkanalytics)
- X/Twitter: [@petroptai](https://x.com/petroptai)
