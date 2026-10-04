from google.cloud import bigquery
import pandas as pd
import numpy as np
from prophet import Prophet
# ------------------------------------------------------------
# 1. Connect to BigQuery
# ------------------------------------------------------------

PROJECT_ID = "gen-lang-client-0915147620"
DATASET_ID = "m5_analytics"

client = bigquery.Client(project=PROJECT_ID)


# ------------------------------------------------------------
# 2. Find high-volume item/store combinations
# ------------------------------------------------------------

top_series_query = f"""
SELECT
    item_id,
    store_id,
    SUM(sales) AS total_sales
FROM `{PROJECT_ID}.{DATASET_ID}.forecast_base`
GROUP BY
    item_id,
    store_id
ORDER BY
    total_sales DESC
LIMIT 3
"""

top_series = client.query(top_series_query).to_dataframe()

print("\nTop 3 high-volume item/store combinations:")
print(top_series)


# ------------------------------------------------------------
# 3. Select the highest-volume series
# ------------------------------------------------------------

selected_item = top_series.iloc[0]["item_id"]
selected_store = top_series.iloc[0]["store_id"]

print(f"\nSelected item: {selected_item}")
print(f"Selected store: {selected_store}")


# ------------------------------------------------------------
# 4. Load only the selected item's history
# ------------------------------------------------------------

forecast_query = f"""
SELECT
    date,
    sales
FROM `{PROJECT_ID}.{DATASET_ID}.forecast_base`
WHERE
    item_id = '{selected_item}'
    AND store_id = '{selected_store}'
ORDER BY
    date
"""

df = client.query(forecast_query).to_dataframe()


# ------------------------------------------------------------
# 5. Prepare data for Prophet
# ------------------------------------------------------------

df["date"] = pd.to_datetime(df["date"])

df = df.rename(
    columns={
        "date": "ds",
        "sales": "y"
    }
)

df = df.sort_values("ds").reset_index(drop=True)


# ------------------------------------------------------------
# 6. Display dataset information
# ------------------------------------------------------------

print("\nProphet dataset prepared successfully")
print(f"Rows: {len(df):,}")
print(f"Date range: {df['ds'].min()} → {df['ds'].max()}")

print("\nFirst 5 rows:")
print(df.head())


# ------------------------------------------------------------
# 6.5. Load M5 calendar events
# ------------------------------------------------------------

calendar_query = f"""
SELECT
    date,
    event_name_1,
    event_name_2
FROM `{PROJECT_ID}.m5_raw.calendar`
WHERE event_name_1 IS NOT NULL
   OR event_name_2 IS NOT NULL
ORDER BY date
"""

calendar_df = client.query(calendar_query).to_dataframe()

calendar_df["date"] = pd.to_datetime(calendar_df["date"])


# ------------------------------------------------------------
# Convert M5 events into Prophet holiday format
# ------------------------------------------------------------

holidays_1 = calendar_df[
    calendar_df["event_name_1"].notna()
][["date", "event_name_1"]].rename(
    columns={
        "date": "ds",
        "event_name_1": "holiday"
    }
)

holidays_2 = calendar_df[
    calendar_df["event_name_2"].notna()
][["date", "event_name_2"]].rename(
    columns={
        "date": "ds",
        "event_name_2": "holiday"
    }
)

holidays = pd.concat(
    [holidays_1, holidays_2],
    ignore_index=True
)

print(f"\nM5 calendar events loaded: {len(holidays):,}")
 
# ------------------------------------------------------------
# 7. Prophet forecasting function
# ------------------------------------------------------------

def forecast_series(item_id, store_id):
    """
    Train and evaluate Prophet for one item/store combination.
    """

    print("\n" + "=" * 60)
    print(f"Processing: {item_id} / {store_id}")
    print("=" * 60)

    # --------------------------------------------------------
    # Load historical sales for this series
    # --------------------------------------------------------

    forecast_query = f"""
    SELECT
        date,
        sales
    FROM `{PROJECT_ID}.{DATASET_ID}.forecast_base`
    WHERE
        item_id = '{item_id}'
        AND store_id = '{store_id}'
    ORDER BY date
    """

    series_df = client.query(forecast_query).to_dataframe()

    series_df["date"] = pd.to_datetime(series_df["date"])

    series_df = series_df.rename(
        columns={
            "date": "ds",
            "sales": "y"
        }
    )

    series_df = series_df.sort_values("ds").reset_index(drop=True)

    print(f"Historical rows: {len(series_df):,}")


    # --------------------------------------------------------
    # Backtesting
    # --------------------------------------------------------

    TEST_DAYS = 30

    train_df = series_df.iloc[:-TEST_DAYS].copy()
    test_df = series_df.iloc[-TEST_DAYS:].copy()

    backtest_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays
    )

    backtest_model.fit(train_df)

    test_future = backtest_model.make_future_dataframe(
        periods=TEST_DAYS,
        freq="D"
    )

    test_forecast = backtest_model.predict(test_future)

    predictions = test_forecast[
        test_forecast["ds"].isin(test_df["ds"])
    ][["ds", "yhat"]].copy()

    predictions["yhat"] = predictions["yhat"].clip(lower=0)


    # --------------------------------------------------------
    # Compare predictions with actual values
    # --------------------------------------------------------

    evaluation = test_df[
        ["ds", "y"]
    ].merge(
        predictions,
        on="ds",
        how="inner"
    )

    mae = np.mean(
        np.abs(
            evaluation["y"] - evaluation["yhat"]
        )
    )

    rmse = np.sqrt(
        np.mean(
            (evaluation["y"] - evaluation["yhat"]) ** 2
        )
    )


    # --------------------------------------------------------
    # Train final model using all historical data
    # --------------------------------------------------------

    final_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays
    )

    final_model.fit(series_df)


    # --------------------------------------------------------
    # Generate 30-day future forecast
    # --------------------------------------------------------

    future = final_model.make_future_dataframe(
        periods=30,
        freq="D"
    )

    forecast = final_model.predict(future)

    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

    future_forecast = forecast.tail(30)[
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ]


    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "item_id": item_id,
        "store_id": store_id,
        "mae": mae,
        "rmse": rmse,
        "forecast": future_forecast
    }


# ------------------------------------------------------------
# 8. Run Prophet for top 3 high-volume series
# ------------------------------------------------------------

results = []

for _, row in top_series.iterrows():

    result = forecast_series(
        item_id=row["item_id"],
        store_id=row["store_id"]
    )

    results.append(result)


# ------------------------------------------------------------
# 9. Display evaluation summary
# ------------------------------------------------------------

evaluation_summary = pd.DataFrame([
    {
        "item_id": result["item_id"],
        "store_id": result["store_id"],
        "MAE": result["mae"],
        "RMSE": result["rmse"]
    }
    for result in results
])

print("\n")
print("=" * 60)
print("PROPHET EVALUATION SUMMARY")
print("=" * 60)

print(
    evaluation_summary.to_string(index=False)
)


# ------------------------------------------------------------
# 10. Display forecast for each series
# ------------------------------------------------------------

for result in results:

    print("\n")
    print("=" * 60)
    print(
        f"30-DAY FORECAST: "
        f"{result['item_id']} / {result['store_id']}"
    )
    print("=" * 60)

    print(
        result["forecast"].to_string(index=False)
    )