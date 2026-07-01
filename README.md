# petro-mcp

<!-- mcp-name: io.github.petropt/petro-mcp -->

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A [Model Context Protocol](https://modelcontextprotocol.io) server for petroleum engineering calculations.

## Install

```bash
pip install petro-mcp
```

## Run

```bash
petro-mcp                              # start the MCP server (stdio)
petro-mcp --list-tools                 # list available tool groups
petro-mcp --tools las,decline          # load only specific groups
petro-mcp --allowed-paths /data/wells  # restrict file reads to a directory
```

## Tool groups

| Group | Tools |
|---|---|
| `las` | Read LAS 2.0 well log files (header, curves, depth-filtered curve data); compare two or more wells |
| `production` | Load production data from CSV (date, well, oil/gas/water rates) |
| `decline` | Arps decline-curve fitting (exponential, hyperbolic, harmonic); EUR |
| `pvt` | Bubble point, black-oil PVT properties, gas Z-factor |
| `petrophysics` | Shale volume, density porosity, Archie water saturation, net pay |
| `reservoir` | Volumetric OOIP/OGIP, recovery factor, P/Z gas material balance |
| `units` | Oilfield unit conversion |

## Dependencies

- `mcp >= 1.0`
- `lasio >= 0.31` (LAS file parsing)
- `numpy >= 1.24`
- `scipy >= 1.10`

## License

MIT.
