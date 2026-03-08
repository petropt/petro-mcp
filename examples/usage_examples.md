# Usage Examples

## Claude Desktop

After configuring petro-mcp in your Claude Desktop config:

### Read a Well Log
> "Read the LAS file at /data/wells/wolfcamp-a1h.las and summarize the available curves"

Claude will use the `read_las` tool to parse the file and return well header info, available curves with statistics, and depth range.

### Analyze Production Decline
> "Load the production data from /data/production.csv for well Wolfcamp A-1H and fit a decline curve"

Claude will:
1. Use `query_production` to load and filter the data
2. Use `fit_decline` to fit exponential, hyperbolic, and harmonic models
3. Compare R-squared values and report the best fit parameters

### Calculate EUR
> "Using the decline parameters from the Wolfcamp A-1H fit, calculate the EUR with a 5 bbl/day economic limit"

Claude will use `calculate_eur` with the fitted qi, Di, and b parameters to estimate ultimate recovery.

### Nodal Analysis
> "Run a nodal analysis for a well with 3500 psi reservoir pressure, PI of 4.2, 2.875-inch tubing, and 150 psi wellhead pressure"

Claude will use `run_nodal_analysis` to find the IPR/VLP intersection and report the expected operating point.

### Calculate Ratios
> "A well is producing 450 bbl/d oil, 1200 Mcf/d gas, and 280 bbl/d water. What are the GOR and water cut?"

Claude will use `calculate_ratios` to compute GOR, WOR, water cut, and classify the well type.

## Cursor / VS Code

The same tools work in Cursor and VS Code with MCP support. Configure the server in your MCP settings and use natural language to interact with well data while coding.

## Programmatic Use

You can also use the underlying functions directly in Python:

```python
from petro_mcp.tools.las import read_las_file
from petro_mcp.tools.decline import fit_decline_curve, calculate_eur

# Read a LAS file
result = read_las_file("/path/to/well.las")

# Fit decline curve
data = [{"time": i, "rate": 1000 * 0.95**i} for i in range(36)]
fit = fit_decline_curve(data, model="hyperbolic")

# Calculate EUR
eur = calculate_eur(qi=1000, Di=0.05, b=1.2, economic_limit=5.0)
```
