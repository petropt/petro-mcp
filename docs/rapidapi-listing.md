# RapidAPI Listing — Petroleum Engineering Calculator API

## Listing Details

| Field               | Value |
|---------------------|-------|
| **API Name**        | Petroleum Engineering Calculator API |
| **Short Description** | Petroleum engineering calculations — DCA, PVT, drilling, petrophysics, economics. Production-grade, unit-tested, zero dependencies on your side. |
| **Category**        | Science > Engineering |
| **Base URL**        | `https://api.petropt.com` |
| **Authentication**  | RapidAPI Proxy (automatic) |

## Pricing Tiers

| Plan       | Price     | Requests/Day | Rate Limit | Best For |
|------------|-----------|--------------|------------|----------|
| **Free**   | $0/mo     | 100          | 10/min     | Evaluation, hobby projects |
| **Basic**  | $19/mo    | 10,000       | 100/min    | Single-well analysis, small teams |
| **Pro**    | $49/mo    | 100,000      | 1,000/min  | Multi-well portfolios, automation |
| **Ultra**  | $99/mo    | Unlimited    | 5,000/min  | Enterprise batch processing |

## Endpoint Groups

### DCA (Decline Curve Analysis)
- `POST /api/v1/decline/fit` — Fit Arps model to production data
- `POST /api/v1/decline/eur` — Calculate Estimated Ultimate Recovery
- `POST /api/v1/decline/forecast` — Generate production forecast

### PVT (Fluid Properties)
- `POST /api/v1/pvt/properties` — Full black-oil PVT suite
- `POST /api/v1/pvt/bubble-point` — Bubble point pressure
- `POST /api/v1/pvt/z-factor` — Gas Z-factor

### Petrophysics
- `POST /api/v1/petrophys/archie` — Archie water saturation
- `POST /api/v1/petrophys/porosity` — Density porosity
- `POST /api/v1/petrophys/vshale` — Shale volume from GR

### Drilling
- `POST /api/v1/drilling/hydrostatic` — Hydrostatic pressure
- `POST /api/v1/drilling/ecd` — Equivalent Circulating Density
- `POST /api/v1/drilling/kill-sheet` — Kill sheet (well control)

### Economics
- `POST /api/v1/economics/npv` — Net Present Value
- `POST /api/v1/economics/well-economics` — Full DCF analysis

### System
- `GET /health` — Health check
- `GET /api/v1/docs` — Endpoint catalog (JSON)

## Example Use Cases

### 1. Fit a decline curve to production data

```bash
curl -X POST "https://petroleum-engineering-calculator-api.p.rapidapi.com/api/v1/decline/fit" \
  -H "Content-Type: application/json" \
  -H "X-RapidAPI-Key: YOUR_API_KEY" \
  -H "X-RapidAPI-Host: petroleum-engineering-calculator-api.p.rapidapi.com" \
  -d '{
    "production_data": [
      {"time": 0, "rate": 1000},
      {"time": 1, "rate": 900},
      {"time": 2, "rate": 820},
      {"time": 3, "rate": 750},
      {"time": 4, "rate": 690},
      {"time": 5, "rate": 640}
    ],
    "model": "hyperbolic"
  }'
```

### 2. Calculate PVT properties

```bash
curl -X POST "https://petroleum-engineering-calculator-api.p.rapidapi.com/api/v1/pvt/properties" \
  -H "Content-Type: application/json" \
  -H "X-RapidAPI-Key: YOUR_API_KEY" \
  -H "X-RapidAPI-Host: petroleum-engineering-calculator-api.p.rapidapi.com" \
  -d '{
    "api_gravity": 35.0,
    "gas_sg": 0.75,
    "temperature": 200.0,
    "pressure": 3000.0,
    "correlation": "standing"
  }'
```

### 3. Water saturation (Archie)

```bash
curl -X POST "https://petroleum-engineering-calculator-api.p.rapidapi.com/api/v1/petrophys/archie" \
  -H "Content-Type: application/json" \
  -H "X-RapidAPI-Key: YOUR_API_KEY" \
  -H "X-RapidAPI-Host: petroleum-engineering-calculator-api.p.rapidapi.com" \
  -d '{
    "rt": 20.0,
    "phi": 0.20,
    "rw": 0.05
  }'
```

### 4. Well economics (full DCF)

```bash
curl -X POST "https://petroleum-engineering-calculator-api.p.rapidapi.com/api/v1/economics/well-economics" \
  -H "Content-Type: application/json" \
  -H "X-RapidAPI-Key: YOUR_API_KEY" \
  -H "X-RapidAPI-Host: petroleum-engineering-calculator-api.p.rapidapi.com" \
  -d '{
    "monthly_oil_bbl": [500, 475, 451, 428, 407, 387],
    "monthly_gas_mcf": [1000, 950, 902, 857, 814, 773],
    "monthly_water_bbl": [50, 50, 50, 50, 50, 50],
    "oil_price_bbl": 75.0,
    "gas_price_mcf": 3.0,
    "opex_monthly": 5000.0,
    "capex": 500000.0
  }'
```

### 5. Kill sheet calculation

```bash
curl -X POST "https://petroleum-engineering-calculator-api.p.rapidapi.com/api/v1/drilling/kill-sheet" \
  -H "Content-Type: application/json" \
  -H "X-RapidAPI-Key: YOUR_API_KEY" \
  -H "X-RapidAPI-Host: petroleum-engineering-calculator-api.p.rapidapi.com" \
  -d '{
    "sidp_psi": 500.0,
    "original_mud_weight_ppg": 10.5,
    "tvd_ft": 12000.0,
    "circulating_pressure_psi": 800.0
  }'
```

## Environment Variables (Server-Side)

| Variable | Description | Default |
|----------|-------------|---------|
| `RAPIDAPI_PROXY_SECRET` | Shared secret from RapidAPI provider dashboard | _(empty — middleware passes through)_ |
| `RAPIDAPI_DEFAULT_DAILY_LIMIT` | Default daily request limit | `100` |

## Rate Limit Response Headers

Every response includes:
- `X-RateLimit-Limit` — daily request allowance for the subscription tier
- `X-RateLimit-Remaining` — requests remaining today
- `X-RateLimit-Reset` — seconds until the limit resets (UTC midnight)

## Deployment Notes

1. Set `RAPIDAPI_PROXY_SECRET` to the value shown in the RapidAPI provider dashboard
2. Point RapidAPI's base URL to your deployment (e.g., `https://api.petropt.com`)
3. Ensure `/health` returns 200 for RapidAPI's uptime monitoring
4. The `/api/v1/docs` endpoint serves as a self-documenting catalog
5. OpenAPI spec is auto-generated at `/openapi.json`
