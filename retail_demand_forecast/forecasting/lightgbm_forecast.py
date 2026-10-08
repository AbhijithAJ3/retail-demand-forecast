"""
LightGBM Demand Forecasting

Purpose:
- Load the top 3 high-volume item/store series from BigQuery
- Create time-series features
- Perform a 30-day recursive backtest
- Train on all historical data
- Generate a 30-day recursive future forecast
- Store forecast outputs in BigQuery
"""

from google.cloud import bigquery
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# 1. BigQuery configuration
# ---------------------------------------------------------

PROJECT_ID = "gen-lang-client-0915147620"

RAW_DATASET = "m5_raw"
ANALYTICS_DATASET = "m5_analytics"

FORECAST_BASE_TABLE = (
    f"{PROJECT_ID}.{ANALYTICS_DATASET}.forecast_base"
)

OUTPUT_TABLE = (
    f"{PROJECT_ID}.{ANALYTICS_DATASET}."
    "lightgbm_forecast_outputs"
)

TEST_DAYS = 30


# Same top 3 series used for Prophet
SERIES = [
    ("FOODS_3_090", "CA_3"),
    ("FOODS_3_586", "TX_2"),
    ("FOODS_3_586", "TX_3"),
]


# ---------------------------------------------------------
# 2. Load historical data
# ---------------------------------------------------------

def load_series(item_id, store_id):

    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
        SELECT
            item_id,
            store_id,
            date,
            sales,
            sell_price,
            year,
            month,
            day_of_week,
            week_of_year
        FROM `{FORECAST_BASE_TABLE}`
        WHERE item_id = @item_id
          AND store_id = @store_id
        ORDER BY date
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "item_id", "STRING", item_id
            ),
            bigquery.ScalarQueryParameter(
                "store_id", "STRING", store_id
            ),
        ]
    )

    df = client.query(
        query,
        job_config=job_config
    ).to_dataframe()

    df["date"] = pd.to_datetime(df["date"])

    return df


# ---------------------------------------------------------
# 3. Load future calendar + price information
# ---------------------------------------------------------

def load_future_data(item_id, store_id, last_date):

    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
        SELECT
            c.date,
            p.sell_price,
            EXTRACT(YEAR FROM c.date) AS year,
            EXTRACT(MONTH FROM c.date) AS month,
            EXTRACT(DAYOFWEEK FROM c.date) AS day_of_week,
            EXTRACT(ISOWEEK FROM c.date) AS week_of_year
        FROM `{PROJECT_ID}.{RAW_DATASET}.calendar` c
        JOIN `{PROJECT_ID}.{RAW_DATASET}.sell_prices` p
            ON c.wm_yr_wk = p.wm_yr_wk
        WHERE p.item_id = @item_id
          AND p.store_id = @store_id
          AND c.date > @last_date
          AND c.date <= DATE_ADD(
              @last_date,
              INTERVAL 30 DAY
          )
        ORDER BY c.date
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "item_id", "STRING", item_id
            ),
            bigquery.ScalarQueryParameter(
                "store_id", "STRING", store_id
            ),
            bigquery.ScalarQueryParameter(
                "last_date", "DATE", last_date.date()
            ),
        ]
    )

    df = client.query(
        query,
        job_config=job_config
    ).to_dataframe()

    df["date"] = pd.to_datetime(df["date"])

    return df


# ---------------------------------------------------------
# 4. Create historical features
# ---------------------------------------------------------

def create_features(df):

    df = df.copy()

    df = df.sort_values("date").reset_index(drop=True)

    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["lag_14"] = df["sales"].shift(14)
    df["lag_28"] = df["sales"].shift(28)

    df["rolling_mean_7"] = (
        df["sales"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    df["rolling_mean_28"] = (
        df["sales"]
        .shift(1)
        .rolling(28)
        .mean()
    )

    df = df.dropna().reset_index(drop=True)

    return df


# ---------------------------------------------------------
# 5. LightGBM model
# ---------------------------------------------------------

FEATURE_COLUMNS = [
    "sell_price",
    "year",
    "month",
    "day_of_week",
    "week_of_year",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
]


def create_model():

    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
    )


# ---------------------------------------------------------
# 6. Recursive forecasting
# ---------------------------------------------------------

def recursive_forecast(
    model,
    history,
    future_data
):
    """
    Predict one future day at a time.

    Each prediction becomes part of the history
    used to generate later lag features.
    """

    history = history.copy()

    predictions = []

    for _, row in future_data.iterrows():

        sales_history = history["sales"].tolist()

        # Historical demand features
        lag_1 = sales_history[-1]
        lag_7 = sales_history[-7]
        lag_14 = sales_history[-14]
        lag_28 = sales_history[-28]

        rolling_mean_7 = np.mean(
            sales_history[-7:]
        )

        rolling_mean_28 = np.mean(
            sales_history[-28:]
        )

        features = pd.DataFrame(
            [{
                "sell_price": row["sell_price"],
                "year": row["year"],
                "month": row["month"],
                "day_of_week": row["day_of_week"],
                "week_of_year": row["week_of_year"],
                "lag_1": lag_1,
                "lag_7": lag_7,
                "lag_14": lag_14,
                "lag_28": lag_28,
                "rolling_mean_7": rolling_mean_7,
                "rolling_mean_28": rolling_mean_28,
            }]
        )

        prediction = model.predict(
            features[FEATURE_COLUMNS]
        )[0]

        # Demand cannot be negative
        prediction = max(float(prediction), 0)

        predictions.append(prediction)

        # Add prediction to history
        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [{
                        "date": row["date"],
                        "sales": prediction
                    }]
                )
            ],
            ignore_index=True
        )

    return np.array(predictions)


# ---------------------------------------------------------
# 7. Process one series
# ---------------------------------------------------------

def process_series(item_id, store_id):

    print("\n" + "=" * 65)
    print(
        f"LightGBM: {item_id} / {store_id}"
    )
    print("=" * 65)

    # ---------------------------------------------
    # Load historical data
    # ---------------------------------------------

    df = load_series(
        item_id,
        store_id
    )

    print(
        f"Loaded {len(df)} historical rows."
    )

    # ---------------------------------------------
    # 30-day backtest
    # ---------------------------------------------

    train_raw = df.iloc[:-TEST_DAYS].copy()
    test_raw = df.iloc[-TEST_DAYS:].copy()

    train_features = create_features(
        train_raw
    )

    model = create_model()

    model.fit(
        train_features[FEATURE_COLUMNS],
        train_features["sales"]
    )

    # Recursive prediction over test period
    test_predictions = recursive_forecast(
        model,
        train_raw[["date", "sales"]],
        test_raw[
            [
                "date",
                "sell_price",
                "year",
                "month",
                "day_of_week",
                "week_of_year",
            ]
        ],
    )

    actual = test_raw["sales"].values

    mae = mean_absolute_error(
        actual,
        test_predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            test_predictions
        )
    )

    print(
        f"Training rows : {len(train_features)}"
    )

    print(
        f"Testing rows  : {len(test_raw)}"
    )

    print(
        f"MAE           : {mae:.2f}"
    )

    print(
        f"RMSE          : {rmse:.2f}"
    )

    # Backtest output
    backtest = test_raw[
        ["item_id", "store_id", "date", "sales"]
    ].copy()

    backtest["predicted_demand"] = (
        test_predictions
    )

    backtest["forecast_type"] = "backtest"

    backtest = backtest.rename(
        columns={
            "sales": "actual_demand"
        }
    )

    # ---------------------------------------------
    # Train final model on all historical data
    # ---------------------------------------------

    full_features = create_features(df)

    final_model = create_model()

    final_model.fit(
        full_features[FEATURE_COLUMNS],
        full_features["sales"]
    )

    # ---------------------------------------------
    # Future 30-day data
    # ---------------------------------------------

    last_date = df["date"].max()

    future_data = load_future_data(
        item_id,
        store_id,
        last_date
    )

    if len(future_data) != 30:

        raise ValueError(
            f"Expected 30 future rows for "
            f"{item_id}/{store_id}, "
            f"but found {len(future_data)}."
        )

    print(
        f"Future rows available: {len(future_data)}"
    )

    # ---------------------------------------------
    # Future forecast
    # ---------------------------------------------

    future_predictions = recursive_forecast(
        final_model,
        df[["date", "sales"]],
        future_data
    )

    future = future_data[
        ["date"]
    ].copy()

    future["item_id"] = item_id
    future["store_id"] = store_id
    future["actual_demand"] = np.nan
    future["predicted_demand"] = (
        future_predictions
    )
    future["forecast_type"] = "future"

    future = future[
        [
            "item_id",
            "store_id",
            "date",
            "actual_demand",
            "predicted_demand",
            "forecast_type",
        ]
    ]

    return (
        backtest,
        future,
        mae,
        rmse
    )


# ---------------------------------------------------------
# 8. Main execution
# ---------------------------------------------------------

if __name__ == "__main__":

    all_outputs = []
    metrics = []

    for item_id, store_id in SERIES:

        backtest, future, mae, rmse = process_series(
            item_id,
            store_id
        )

        all_outputs.append(backtest)
        all_outputs.append(future)

        metrics.append(
            {
                "item_id": item_id,
                "store_id": store_id,
                "mae": mae,
                "rmse": rmse,
            }
        )

    # ---------------------------------------------
    # Combine all outputs
    # ---------------------------------------------

    output_df = pd.concat(
        all_outputs,
        ignore_index=True
    )

    output_df["forecast_date"] = pd.to_datetime(
        output_df["date"]
    ).dt.date

    output_df = output_df.drop(
        columns=["date"]
    )

    # ---------------------------------------------
    # Display summary
    # ---------------------------------------------

    metrics_df = pd.DataFrame(
        metrics
    )

    print("\n")
    print("=" * 65)
    print("LIGHTGBM SUMMARY")
    print("=" * 65)

    print(
        metrics_df.to_string(index=False)
    )

    print("\n")
    print(
        f"Total output rows: {len(output_df)}"
    )

    print(
        output_df["forecast_type"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------
    # Upload to BigQuery
    # ---------------------------------------------

    client = bigquery.Client(
        project=PROJECT_ID
    )

    schema = [
        bigquery.SchemaField(
            "item_id",
            "STRING"
        ),
        bigquery.SchemaField(
            "store_id",
            "STRING"
        ),
        bigquery.SchemaField(
            "forecast_date",
            "DATE"
        ),
        bigquery.SchemaField(
            "actual_demand",
            "FLOAT"
        ),
        bigquery.SchemaField(
            "predicted_demand",
            "FLOAT"
        ),
        bigquery.SchemaField(
            "forecast_type",
            "STRING"
        ),
    ]

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=(
            bigquery.WriteDisposition.WRITE_TRUNCATE
        )
    )

    load_job = client.load_table_from_dataframe(
        output_df,
        OUTPUT_TABLE,
        job_config=job_config
    )

    load_job.result()

    print("\n")
    print("=" * 65)
    print("BIGQUERY UPLOAD COMPLETE")
    print("=" * 65)

    print(
        f"Table: {OUTPUT_TABLE}"
    )

    print(
        f"Rows uploaded: {len(output_df)}"
    )