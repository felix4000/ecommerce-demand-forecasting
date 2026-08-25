"""
forecasting.py

Weekly demand forecasting by product category using Holt-Winters
exponential smoothing (additive trend + seasonal), with a holdout period
to measure forecast accuracy (MAPE) rather than just plotting a fitted
line. This mirrors the demand-forecasting work referenced in my
professional experience (AI-assisted stock/demand forecasting), applied
here to synthetic weekly sales data.

Usage:
    python python/forecasting.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
HOLDOUT_WEEKS = 8
SEASONAL_PERIODS = 52


def load_demand():
    df = pd.read_csv(os.path.join(DATA_DIR, "weekly_demand_by_category.csv"), parse_dates=["week"])
    return df


def mape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


def forecast_category(series: pd.Series, holdout: int = HOLDOUT_WEEKS):
    """Fits Holt-Winters on all but the last `holdout` weeks, forecasts
    that holdout, and returns (mape, forecast_next_period, fitted_model)."""
    train = series.iloc[:-holdout]
    test = series.iloc[-holdout:]

    # short series (< 2 full seasonal cycles) fall back to trend-only smoothing
    seasonal = "add" if len(train) >= 2 * SEASONAL_PERIODS else None
    sp = SEASONAL_PERIODS if seasonal else None

    model = ExponentialSmoothing(
        train, trend="add", seasonal=seasonal, seasonal_periods=sp, initialization_method="estimated"
    ).fit()

    holdout_forecast = model.forecast(holdout)
    accuracy = mape(test.values, holdout_forecast.values)

    # refit on full series for the actual forward forecast
    full_model = ExponentialSmoothing(
        series, trend="add", seasonal=seasonal, seasonal_periods=sp, initialization_method="estimated"
    ).fit()
    next_forecast = full_model.forecast(4)  # next 4 weeks

    return accuracy, next_forecast, model


def main():
    demand = load_demand()
    results = []
    for category, g in demand.groupby("category"):
        g = g.sort_values("week").set_index("week")["units_sold"]
        g.index = pd.DatetimeIndex(g.index).to_period("W").to_timestamp()
        accuracy, next_forecast, _ = forecast_category(g)
        results.append({
            "category": category,
            "mape_pct": round(accuracy, 1),
            "avg_weekly_units": round(g.mean(), 0),
            "next_4_week_forecast": [round(v, 0) for v in next_forecast.values],
        })

    results_df = pd.DataFrame(results).sort_values("mape_pct")
    print("=== Forecast accuracy by category (holdout MAPE, lower is better) ===")
    print(results_df.to_string(index=False))

    print(f"\nAverage MAPE across categories: {results_df['mape_pct'].mean():.1f}%")


if __name__ == "__main__":
    main()
