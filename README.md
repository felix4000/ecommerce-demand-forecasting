# E-commerce Demand Forecasting

Weekly product-category demand forecasting with Holt-Winters exponential
smoothing, measured honestly with a holdout period rather than reported as
a fitted-line chart. Built to demonstrate the same kind of AI-assisted
demand/stock forecasting referenced in my professional experience.

> This project uses synthetic/anonymised data inspired by real-world
> e-commerce demand forecasting scenarios. No confidential company data,
> real stock levels or real demand figures are included. See
> [docs/methodology.md](docs/methodology.md).

## Business Problem

Stock planning needs a forecast, but a forecast without a measured accuracy
is just an opinion with a chart. The brief here: forecast weekly demand per
category, and report how good that forecast actually is before it gets used
to plan stock.

## Objectives

- Forecast weekly demand per product category.
- Measure forecast accuracy (MAPE) on a genuine holdout, not fitted error.
- Identify which categories forecast reliably and which don't, and why.

## Dataset

`data/weekly_demand_by_category.csv` — 416 rows, 8 categories x 52 weeks.
Full detail: [docs/methodology.md](docs/methodology.md).

## Methodology

1. Split each category's series into train and an 8-week holdout.
2. Fit Holt-Winters (trend component; see methodology for why not seasonal).
3. Forecast the holdout, score with MAPE.
4. Refit on the full series, produce the forward 4-week forecast.
5. Compare accuracy across categories to inform safety-stock policy.

## Data Architecture

```text
Daily/weekly sales  ──►  Weekly demand by category  ──►  Train / holdout split
                                                                  │
                                                                  ▼
                                                     Holt-Winters (trend)
                                                                  │
                                                    ┌─────────────┴─────────────┐
                                                    ▼                           ▼
                                        Holdout MAPE (accuracy)      Forward 4-week forecast
                                                    │
                                                    ▼
                                       Category-specific safety-stock policy
```

## Tools

Python, pandas, statsmodels (Holt-Winters exponential smoothing), Jupyter.
Professional experience: Azure ML for demand/stock forecasting at
Euro4x4parts and Groupe Cipanguo — see [profile](https://github.com/felix4000).

## Analysis

| Area | File |
|---|---|
| Forecasting script | [`python/forecasting.py`](python/forecasting.py) |
| Full walkthrough notebook | [`notebooks/demand_forecasting.ipynb`](notebooks/demand_forecasting.ipynb) |

```bash
$ python python/forecasting.py
=== Forecast accuracy by category (holdout MAPE, lower is better) ===
       category  mape_pct  avg_weekly_units
   Transmission       9.0              21.0
     Suspension       9.2              91.0
Body & Exterior      10.6              58.0
        Filters      12.0             221.0
 Cooling System      13.6              50.0
 Braking System      13.9             132.0
       Lighting      39.4              72.0
   Engine Parts      39.7              44.0

Average MAPE across categories: 18.4%
```

## Key Findings

1. **Forecast accuracy ranges from 9% to 40% MAPE across categories** —
   averaging them into a single number would hide a real, actionable split.
2. **High-volume categories (Transmission, Suspension) forecast far more
   reliably than lower-volume, noisier ones (Engine Parts, Lighting).**
3. **One year of history isn't enough to fit a genuine seasonal component** —
   documented as a limitation rather than forced into an unreliable model.

## Recommendations

| Finding | Recommendation |
|---|---|
| MAPE varies 9-40% by category | Set safety-stock buffers per category based on its own forecast accuracy, not one blanket rule |
| Low-volume categories forecast worst | Consider a simpler moving-average approach for these — a complex model isn't earning its complexity here |
| 1 year of history limits seasonality | Prioritise collecting 18-24 months of clean history before attempting a seasonal model |

## Project Structure

```text
ecommerce-demand-forecasting/
├── README.md
├── data/
│   └── weekly_demand_by_category.csv
├── python/
│   └── forecasting.py
├── notebooks/
│   └── demand_forecasting.ipynb
└── docs/
    └── methodology.md
```

## How to Run

```bash
pip install pandas numpy statsmodels
python python/forecasting.py
jupyter notebook notebooks/demand_forecasting.ipynb
```

## Limitations

See [docs/methodology.md](docs/methodology.md) — trend-only model (no
reliable seasonal component with 1 year of history), category-level rather
than SKU-level, no promotional/stockout flags in the synthetic data.

## About the Author

**Felix Ibeh** — Data Analyst. Predictive analytics and AI-assisted demand
forecasting (Azure ML) as part of my work at Groupe Cipanguo and
Euro4x4parts.

[LinkedIn](https://www.linkedin.com/in/felix-ibeh-data-analyst/) ·
[CV](https://felix4000.github.io/felix-ibeh-cv/) ·
[GitHub](https://github.com/felix4000)
# ecommerce-demand-forecasting
Weekly demand forecasting with Holt-Winters exponential smoothing, measured with a holdout (synthetic data)
