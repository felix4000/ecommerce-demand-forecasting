# Methodology

## Data

`data/weekly_demand_by_category.csv` — synthetic weekly units sold per
product category, 52 weeks (Aug 2025 - Jul 2026), 8 categories, 416 rows.
Built with a mild upward trend, an annual seasonal sine wave, and random
noise per category — the same kind of series a stock-planning or
category-management extract would produce.

> This project uses synthetic/anonymised data inspired by real-world
> e-commerce demand forecasting scenarios. No confidential company data,
> real stock levels or real demand figures are included.

## Method

Holt-Winters exponential smoothing (`statsmodels.tsa.holtwinters.ExponentialSmoothing`),
trend component only. An 8-week holdout is used to measure forecast
accuracy (MAPE — mean absolute percentage error) before producing the
forward 4-week forecast, so the accuracy number is honest: it is measured
on weeks the model did not see, not fitted-value error.

## Why trend-only, not full seasonal Holt-Winters

52 weeks of history is one full annual cycle — Holt-Winters needs at least
two full cycles to estimate a seasonal component reliably. Forcing a
seasonal fit on one year of data would produce a seasonal index the model
is effectively memorising from a single instance, not a generalisable
pattern. The forecast here is trend-only for that reason, and the
limitation is intentional rather than hidden. With 18-24 months of history,
the same script (swap `seasonal=None` for `seasonal="add"`, `seasonal_periods=52`
in `forecast_category()`) would fit a real seasonal component.

## Accuracy

MAPE ranges from about 9% (Transmission, Suspension) to about 40%
(Engine Parts, Lighting) across categories, averaging 18.4%. The gap is the
finding: low-volume, higher-noise categories forecast worse, and that
should inform how much buffer stock gets carried for them rather than
applying one blanket safety-stock rule across the catalogue.

## Tools

Python, pandas, statsmodels (Holt-Winters exponential smoothing). The same
logic is what sits behind AI-assisted demand forecasting in a production
setting (my professional experience: Azure ML for stock-forecasting models
at Euro4x4parts and Groupe Cipanguo — see [profile](https://github.com/felix4000)).

## Limitations

- One year of history: no reliable seasonal component (see above).
- No promotional or stockout flags in the synthetic data — a real
  forecasting pipeline needs to separate "no demand" from "no stock to sell",
  which this dataset doesn't need to since it has no stockouts by
  construction.
- Category-level, not SKU-level — SKU-level forecasting needs materially
  more history per series to avoid overfitting.
