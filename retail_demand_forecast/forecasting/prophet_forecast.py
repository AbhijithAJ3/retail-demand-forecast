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

#---------------------------------------------------
# 2. Select representative product/store combinations
# ------------------------------------------------------------

SERIES = [
    # FOODS
    ("FOODS_3_090", "CA_3"),
    ("FOODS_3_586", "TX_2"),
    ("FOODS_3_586", "TX_3"),

    # HOBBIES
    ("HOBBIES_1_001", "CA_1"),
    ("HOBBIES_1_001", "TX_1"),
    ("HOBBIES_1_002", "CA_1"),

    # HOUSEHOLD
    ("HOUSEHOLD_1_001", "CA_1"),
    ("HOUSEHOLD_1_001", "TX_1"),
    ("HOUSEHOLD_1_002", "CA_1"),
]

print(f"Selected {len(SERIES)} product/store combinations:")
for item_id, store_id in SERIES:
    print(f"  {item_id} / {store_id}")



# ------------------------------------------------------------
# 3. Display selected series
# ------------------------------------------------------------
print("\nProphet will process all selected series.")

 

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
    "evaluation": evaluation,
    "forecast": future_forecast
    }



# ------------------------------------------------------------
# 8. Run Prophet for all selected product/store combinations
# ------------------------------------------------------------

results = []

for item_id, store_id in SERIES:
    result = forecast_series(
        item_id=item_id,
        store_id=store_id
    )

    results.append(result)


# ------------------------------------------------------------
# 8. Combine forecast outputs
# ------------------------------------------------------------

forecast_outputs = []

for result in results:

    # Backtest predictions with actual demand
    backtest_output = result["evaluation"].copy()

    backtest_output["item_id"] = result["item_id"]
    backtest_output["store_id"] = result["store_id"]
    backtest_output["forecast_type"] = "backtest"

    backtest_output = backtest_output.rename(
        columns={
            "y": "actual_demand",
            "yhat": "predicted_demand"
        }
    )

    backtest_output["lower_bound"] = np.nan
    backtest_output["upper_bound"] = np.nan


    # Future 30-day forecast
    future_output = result["forecast"].copy()

    future_output["item_id"] = result["item_id"]
    future_output["store_id"] = result["store_id"]
    future_output["forecast_type"] = "future"

    future_output = future_output.rename(
        columns={
            "yhat": "predicted_demand",
            "yhat_lower": "lower_bound",
            "yhat_upper": "upper_bound"
        }
    )

    future_output["actual_demand"] = np.nan


    # Keep the same columns
    backtest_output = backtest_output[
        [
            "item_id",
            "store_id",
            "ds",
            "actual_demand",
            "predicted_demand",
            "lower_bound",
            "upper_bound",
            "forecast_type"
        ]
    ]

    future_output = future_output[
        [
            "item_id",
            "store_id",
            "ds",
            "actual_demand",
            "predicted_demand",
            "lower_bound",
            "upper_bound",
            "forecast_type"
        ]
    ]


    forecast_outputs.append(backtest_output)
    forecast_outputs.append(future_output)


forecast_outputs = pd.concat(
    forecast_outputs,
    ignore_index=True
)

print("\nForecast output dataset prepared")
print(f"Rows: {len(forecast_outputs):,}")

print("\nOutput columns:")
print(forecast_outputs.columns.tolist())

print("\nForecast type counts:")
print(
    forecast_outputs["forecast_type"]
    .value_counts()
)

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

# ------------------------------------------------------------
# 11. Store Prophet forecast outputs in BigQuery
# ------------------------------------------------------------

OUTPUT_TABLE = f"{PROJECT_ID}.{DATASET_ID}.prophet_forecast_outputs"

# Rename Prophet date column for clarity
forecast_outputs = forecast_outputs.rename(
    columns={"ds": "forecast_date"}
)

# Make sure date is stored as a date
forecast_outputs["forecast_date"] = pd.to_datetime(
    forecast_outputs["forecast_date"]
).dt.date


# Define BigQuery schema
schema = [
    bigquery.SchemaField("item_id", "STRING"),
    bigquery.SchemaField("store_id", "STRING"),
    bigquery.SchemaField("forecast_date", "DATE"),
    bigquery.SchemaField("actual_demand", "FLOAT"),
    bigquery.SchemaField("predicted_demand", "FLOAT"),
    bigquery.SchemaField("lower_bound", "FLOAT"),
    bigquery.SchemaField("upper_bound", "FLOAT"),
    bigquery.SchemaField("forecast_type", "STRING"),
]


# Configure BigQuery load
job_config = bigquery.LoadJobConfig(
    schema=schema,
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
)


print("\nUploading forecast outputs to BigQuery...")

load_job = client.load_table_from_dataframe(
    forecast_outputs,
    OUTPUT_TABLE,
    job_config=job_config
)

load_job.result()

print("Forecast outputs uploaded successfully!")
print(f"Table: {OUTPUT_TABLE}")


# Verify table
table = client.get_table(OUTPUT_TABLE)

print(f"Rows stored in BigQuery: {table.num_rows}")
print(f"Columns stored: {len(table.schema)}")