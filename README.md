# petro-mcp

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple.svg)](https://modelcontextprotocol.io)

**MCP server that gives LLMs access to 70 petroleum engineering tools.**

Parse well logs, fit decline curves, run PVT correlations, calculate drilling hydraulics, evaluate well economics, and more -- all through natural language with any MCP-compatible AI assistant.

Built by [Groundwork Analytics](https://petropt.com)

---

## What is this?

petro-mcp is a [Model Context Protocol](https://modelcontextprotocol.io) server that exposes petroleum engineering workflows to LLMs like Claude and other MCP-compatible assistants. Instead of writing scripts to parse LAS files or fit decline curves, just ask your AI assistant in plain English.

**Why MCP?** MCP is an open standard that lets AI assistants interact with external data and tools. While other energy MCP servers provide commodity prices ([OilpriceAPI](https://github.com/OilpriceAPI/mcp-server)), petro-mcp is purpose-built for petroleum engineering workflows -- well log interpretation, decline curve analysis, reservoir engineering, drilling calculations, production engineering, and economics.

**Prefer a web UI?** Try the [Production Close Assistant](https://tools.petropt.com/production-close/) -- upload messy production data, get clean analysis-ready output in seconds. Free, no signup. Also: [Decline Curve Analysis](https://tools.petropt.com/decline-curve/) | [Well Economics](https://tools.petropt.com/well-economics/)

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

For well trajectory tools (optional):

```bash
pip install petro-mcp[trajectory]
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

### Selective tool loading

Load only the tool groups you need to reduce token overhead:

```bash
petro-mcp --tools decline,pvt,economics
petro-mcp --list-tools  # show available groups
```

Available groups: `las`, `production`, `decline`, `pvt`, `petrophysics`, `calculations`, `units`, `drilling`, `reservoir`, `economics`, `production_eng`, `trajectory`.

### Use it

Once configured, ask your AI assistant:

- *"Read the LAS file at /data/wells/wolfcamp.las and summarize the curves"*
- *"Load production data from /data/prod.csv and fit a decline curve for Wolfcamp A-1H"*
- *"Calculate EUR with qi=800, Di=0.06, b=1.1 and a 5 bbl/day economic limit"*
- *"Run a nodal analysis: 3500 psi reservoir, PI=4.2, 2.875-inch tubing, 150 psi wellhead"*
- *"Calculate PVT properties for 35 API oil at 200F and 3000 psi"*
- *"What's the kill mud weight if SIDP is 500 psi, current MW is 10.5 ppg, TVD 12000 ft?"*

## Tools (70)

### Well Logs (4 tools)

| Tool | Description |
|------|-------------|
| `read_las` | Parse a LAS 2.0 file -- returns well header and curve data summary |
| `get_header` | Extract well header metadata (name, UWI, location, KB, TD) |
| `get_curves` | List all curves with units and descriptions |
| `get_curve_values` | Get curve data with optional depth range filtering |

### Production Data & Trends (2 tools)

| Tool | Description |
|------|-------------|
| `query_production` | Query production CSV data by well name and date range |
| `analyze_trends` | Detect production anomalies: shut-ins, rate jumps, water breakthrough, GOR blowouts |

### Decline Curve Analysis (6 tools)

| Tool | Description |
|------|-------------|
| `fit_decline` | Fit Arps decline curves (exponential, hyperbolic, harmonic, modified hyperbolic, Duong) |
| `calculate_eur` | Calculate Estimated Ultimate Recovery from decline parameters |
| `fit_ple_decline` | Fit Power Law Exponential (PLE) decline model (Ilk et al., 2008) |
| `fit_duong_decline` | Fit Duong decline model for fracture-dominated flow (Duong, 2011) |
| `fit_sepd_decline` | Fit Stretched Exponential (SEPD) decline model (Valko, 2009) |
| `forecast_advanced_decline` | Forecast production using PLE, Duong, SEPD, or THM models |

### PVT Correlations (5 tools)

| Tool | Description |
|------|-------------|
| `calculate_pvt_properties` | Black-oil PVT: Pb, Bo, Rs, viscosity, Z-factor (Standing, Vasquez-Beggs, Petrosky-Farshad) |
| `calculate_bubble_point` | Bubble point pressure from Standing's correlation |
| `calculate_oil_co` | Oil compressibility above and below bubble point (Vasquez-Beggs) |
| `calculate_brine_pvt` | Brine PVT: density, viscosity, FVF, compressibility (McCain, Osif) |
| `calculate_gas_z` | Gas Z-factor with Hall-Yarborough or Dranchuk-Abou-Kassem; Sutton or Piper pseudocriticals |

### Petrophysics (12 tools)

| Tool | Description |
|------|-------------|
| `calculate_vshale` | Shale volume from gamma ray (linear, Larionov, Clavier) |
| `calculate_density_porosity` | Porosity from bulk density log |
| `calculate_sonic_porosity` | Porosity from sonic log (Wyllie, Raymer-Hunt-Gardner) |
| `calculate_nd_porosity` | Quick-look neutron-density porosity (RMS method) |
| `calculate_effective_porosity` | Effective porosity from total porosity and shale volume |
| `calculate_archie_sw` | Water saturation using Archie equation (clean sands) |
| `calculate_simandoux_sw` | Water saturation using Simandoux equation (shaly sands) |
| `calculate_indonesian_sw` | Water saturation using Indonesian equation (high-Vshale formations) |
| `calculate_permeability_timur` | Permeability estimate using Timur (1968) |
| `calculate_permeability_coates` | Permeability estimate using Coates (1991) |
| `calculate_net_pay` | Net pay determination with porosity, Sw, and Vshale cutoffs |
| `calculate_hpt` | Hydrocarbon pore thickness (HPT) |

### Reservoir Engineering (6 tools)

| Tool | Description |
|------|-------------|
| `pz_analysis` | Gas material balance: P/Z vs cumulative gas production (OGIP estimation) |
| `havlena_odeh` | Oil material balance: Havlena-Odeh straight-line method (drive mechanism ID) |
| `volumetric_ooip` | Volumetric Original Oil In Place |
| `volumetric_ogip` | Volumetric Original Gas In Place |
| `recovery_factor` | Recovery factor from cumulative production and OOIP/OGIP |
| `radius_of_investigation` | Radius of investigation for well tests (Lee, 1982) |

### Drilling Engineering (11 tools)

| Tool | Description |
|------|-------------|
| `calculate_hydrostatic_pressure` | Hydrostatic pressure (P = 0.052 * MW * TVD) |
| `calculate_ecd` | Equivalent circulating density |
| `calculate_formation_pressure_gradient` | Formation pressure gradient as ppg equivalent |
| `calculate_kill_mud_weight` | Kill mud weight for well control |
| `calculate_icp_fcp` | Initial and Final Circulating Pressures (Driller's method) |
| `calculate_maasp` | Maximum Allowable Annular Surface Pressure |
| `calculate_annular_velocity` | Annular velocity |
| `calculate_nozzle_tfa` | Total flow area of bit nozzles |
| `calculate_bit_pressure_drop` | Pressure drop across the bit |
| `calculate_burst_pressure` | Internal burst pressure (Barlow with API tolerance) |
| `calculate_collapse_pressure` | Collapse pressure rating per API 5C3 |

### Production Engineering (7 tools)

| Tool | Description |
|------|-------------|
| `calculate_beggs_brill` | Beggs & Brill (1973) multiphase pressure drop in pipes |
| `calculate_turner_critical` | Turner et al. (1969) critical rate for gas well liquid unloading |
| `calculate_coleman_critical` | Coleman et al. (1991) critical rate for liquid loading |
| `calculate_hydrate_temp` | Hydrate formation temperature (gas-gravity method) |
| `calculate_hydrate_inhibitor` | Hydrate inhibitor dosing (Hammerschmidt equation) |
| `calculate_erosional_vel` | Erosional velocity per API RP 14E |
| `calculate_choke_flow` | Choke flow rate using Gilbert correlation (1954) |

### Economics (8 tools)

| Tool | Description |
|------|-------------|
| `calculate_well_economics` | Full DCF analysis: NPV, IRR, payout, profitability index |
| `calculate_npv` | Net Present Value from monthly cash flows |
| `calculate_irr` | Internal Rate of Return via bisection |
| `calculate_pv10` | PV10 (SEC standard present value at 10% discount) |
| `calculate_breakeven_price` | Breakeven oil price (minimum price for NPV = 0) |
| `calculate_operating_netback` | Operating netback per BOE |
| `calculate_payout_period` | Payout period from monthly cash flows |
| `calculate_price_sensitivity` | NPV across multiple price scenarios for sensitivity analysis |

### Nodal Analysis & Ratios (2 tools)

| Tool | Description |
|------|-------------|
| `calculate_ratios` | Producing ratios: GOR, WOR, water cut; well type classification |
| `run_nodal_analysis` | IPR/VLP intersection for operating point (Vogel IPR) |

### Unit Conversions (2 tools)

| Tool | Description |
|------|-------------|
| `convert_oilfield_units` | Convert between oilfield and SI units (pressure, volume, density, temperature, etc.) |
| `list_oilfield_units` | List all supported unit categories and conversions |

### Well Trajectory (5 tools, optional)

Requires `pip install petro-mcp[trajectory]` (uses [welleng](https://github.com/jonnymaserati/welleng)).

| Tool | Description |
|------|-------------|
| `calculate_well_survey` | Well trajectory using minimum curvature method |
| `calculate_dogleg_severity` | Dogleg severity between two survey stations |
| `calculate_vertical_section` | Project trajectory onto a vertical section plane |
| `calculate_wellbore_tortuosity` | Wellbore tortuosity index from survey data |
| `check_well_anticollision` | Separation check between two wells at closest approach |

### Decline Curve Analysis

The decline curve tools are physics-constrained:

- **b-factor** bounded to [0, 2] (physical range for Arps models)
- **Non-negative rates** enforced throughout
- **Six models**: exponential, hyperbolic, harmonic, modified hyperbolic, Duong, PLE, SEPD
- Advanced models (PLE, Duong, SEPD, THM) use [petbox-dca](https://github.com/petbox-dev/dca) for the forward model
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
Depth range: 5000-5020 ft, 0.5 ft step (41 samples)

+-------+------+----------------------------+---------------+
| Curve | Unit |        Description         |     Range     |
+-------+------+----------------------------+---------------+
| GR    | GAPI | Gamma Ray                  | 22-126        |
| RHOB  | G/CC | Bulk Density               | 2.53-2.67     |
| NPHI  | V/V  | Neutron Porosity           | 0.09-0.23     |
| ILD   | OHMM | Deep Induction Resistivity | 3.9-175       |
| SP    | MV   | Spontaneous Potential      | -30.5 to -2.0 |
+-------+------+----------------------------+---------------+

Quick interpretation:
- Shale zones (~5002-5005 ft): High GR (>100), low resistivity,
  high NPHI -- classic shale signature.
- Clean/tight zones (~5008-5010 ft): Low GR (<30), high resistivity
  (>100 ohm-m), low NPHI -- likely tight carbonate, potentially
  hydrocarbon-bearing.
```

## Hosted API (RapidAPI)

Don't want to run the server yourself? Use the hosted API -- free tier available, no setup required.

**[Subscribe on RapidAPI](https://rapidapi.com/groundwork-analytics-groundwork-analytics-default/api/petro-mcp)** -- 16 endpoints for decline curves, PVT, petrophysics, drilling, and economics.

| Plan | Price | Requests/month |
|------|-------|----------------|
| Free | $0 | 1,000 |
| Pro | $49/mo | 100,000 |
| Ultra | $99/mo | 1,000,000 |

```bash
curl -X POST "https://petro-mcp.p.rapidapi.com/api/v1/decline/fit" \
  -H "X-RapidAPI-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"production_data":[{"time":0,"rate":1000},{"time":1,"rate":850},{"time":2,"rate":730}]}'
```

Also available as interactive web tools at [tools.petropt.com](https://tools.petropt.com).

## REST API (Self-Hosted)

petro-mcp includes a FastAPI service that exposes the calculation engines as REST endpoints. This powers [tools.petropt.com](https://tools.petropt.com) -- the web frontend calls these endpoints instead of re-implementing formulas in JavaScript.

### Quick start

```bash
pip install petro-mcp
petro-api              # starts on http://localhost:8000
# or: uvicorn petro_mcp.api.app:app --reload
```

### Endpoints

| Group | Endpoint | Description |
|-------|----------|-------------|
| DCA | `POST /api/v1/decline/fit` | Fit Arps decline curve to production data |
| DCA | `POST /api/v1/decline/eur` | Calculate EUR from decline parameters |
| DCA | `POST /api/v1/decline/forecast` | Generate production forecast |
| PVT | `POST /api/v1/pvt/properties` | Black-oil PVT properties (Standing, Vasquez-Beggs, Petrosky-Farshad) |
| PVT | `POST /api/v1/pvt/bubble-point` | Bubble point pressure |
| PVT | `POST /api/v1/pvt/z-factor` | Gas Z-factor (Hall-Yarborough, Dranchuk) |
| Petrophysics | `POST /api/v1/petrophys/archie` | Water saturation (Archie) |
| Petrophysics | `POST /api/v1/petrophys/porosity` | Density porosity |
| Petrophysics | `POST /api/v1/petrophys/vshale` | Shale volume from gamma ray |
| Drilling | `POST /api/v1/drilling/hydrostatic` | Hydrostatic pressure |
| Drilling | `POST /api/v1/drilling/ecd` | Equivalent circulating density |
| Drilling | `POST /api/v1/drilling/kill-sheet` | Kill mud weight + ICP/FCP |
| Economics | `POST /api/v1/economics/npv` | Net Present Value |
| Economics | `POST /api/v1/economics/well-economics` | Full DCF analysis |

Interactive docs at `http://localhost:8000/docs` (Swagger UI).

### Example

```bash
curl -X POST http://localhost:8000/api/v1/decline/eur \
  -H "Content-Type: application/json" \
  -d '{"qi": 800, "Di": 0.06, "b": 1.2}'
```

## Limitations

- **LAS 2.0 only** -- LAS 3.0 and DLIS formats are not yet supported
- **Simplified nodal analysis** -- Uses Vogel IPR and a simplified friction model. Results are directional/screening-level, not suitable for detailed well design
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
- Well spacing and interference analysis
- Additional multiphase flow correlations

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
