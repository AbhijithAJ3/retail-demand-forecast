
import streamlit as st
import pandas as pd
from google.cloud import bigquery

# Dashboard configuration
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

st.title("Retail Demand Forecasting & Inventory Optimization")
st.caption("M5 Walmart dataset | Prophet and LightGBM forecasts")

PROJECT_ID = "gen-lang-client-0915147620"
DATASET_ID = "m5_analytics"

# Connect to BigQuery using existing Google Cloud credentials
@st.cache_resource
def get_client():
    return bigquery.Client(project=PROJECT_ID)

client = get_client()


# Load forecast outputs
@st.cache_data(ttl=300)
def load_forecasts(table_name):
    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ID}.{table_name}`
    """
    return client.query(query).to_dataframe()


# Load product category and department mappings
@st.cache_data(ttl=300)
def load_product_mapping():
    query = f"""
        SELECT
            item_id,
            ANY_VALUE(cat_id) AS cat_id,
            ANY_VALUE(dept_id) AS dept_id
        FROM `{PROJECT_ID}.{DATASET_ID}.forecast_base`
        GROUP BY item_id
    """
    return client.query(query).to_dataframe()


try:
    # Load both forecasting models
    prophet_df = load_forecasts("prophet_forecast_outputs")
    lightgbm_df = load_forecasts("lightgbm_forecast_outputs")
    product_mapping = load_product_mapping()

    # Normalize forecast dates
    for forecast_df in [prophet_df, lightgbm_df]:
        forecast_df["forecast_date"] = pd.to_datetime(
            forecast_df["forecast_date"]
        )

    # Model selection
    model = st.selectbox(
        "Choose forecasting model",
        ["LightGBM", "Prophet"]
    )

    df = lightgbm_df if model == "LightGBM" else prophet_df
    # Prepare backtest data for model evaluation
    if "forecast_type" in df.columns:
        backtest_df = df[
            df["forecast_type"].astype(str).str.lower() == "backtest"
        ].copy()
    else:
        backtest_df = pd.DataFrame()

    # Ensure demand columns contain numeric values
    if not backtest_df.empty:
        backtest_df["actual_demand"] = pd.to_numeric(
            backtest_df["actual_demand"], errors="coerce"
        )
        backtest_df["predicted_demand"] = pd.to_numeric(
            backtest_df["predicted_demand"], errors="coerce"
        )

    # Show future forecasts only
    if "forecast_type" in df.columns:
        future_df = df[
            df["forecast_type"].astype(str).str.lower() == "future"
        ].copy()
    else:
        future_df = df.copy()

    if future_df.empty:
        st.warning("No future forecast rows were found.")
        st.stop()

    # Attach category and department to each forecast row
    future_df = future_df.merge(
        product_mapping,
        on="item_id",
        how="left",
        validate="many_to_one"
    )

    if future_df[["cat_id", "dept_id"]].isna().any().any():
        st.warning(
            "Some products could not be mapped to a category "
            "or department. Check forecast_base."
        )

    # Show only categories with available forecast outputs
    categories = sorted(future_df["cat_id"].dropna().unique())

    if not categories:
        st.error("No category mappings are available for these forecasts.")
        st.stop()

    st.subheader("Forecast Filters")

    selected_category = st.selectbox(
        "Select category",
        categories
    )

    category_df = future_df[
        future_df["cat_id"] == selected_category
    ].copy()

    departments = sorted(category_df["dept_id"].dropna().unique())

    if not departments:
        st.warning("No departments have forecasts in this category.")
        st.stop()

    selected_department = st.selectbox(
        "Select department",
        departments
    )

    department_df = category_df[
        category_df["dept_id"] == selected_department
    ].copy()

    products = sorted(department_df["item_id"].dropna().unique())

    if not products:
        st.warning("No products have forecasts in this department.")
        st.stop()

    selected_item = st.selectbox(
        "Select product",
        products
    )

    product_df = department_df[
        department_df["item_id"] == selected_item
    ].copy()

    stores = sorted(product_df["store_id"].dropna().unique())

    if not stores:
        st.warning("No stores have forecasts for this product.")
        st.stop()

    selected_store = st.selectbox(
        "Select store",
        stores
    )

    # Filter selected product and store
    filtered = product_df[
        product_df["store_id"] == selected_store
    ].sort_values("forecast_date").copy()

    if filtered.empty:
        st.warning("No forecast data matches these selections.")
        st.stop()

    # Forecast summary
    st.subheader("Forecast Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Product", selected_item)
    col2.metric("Store", selected_store)
    col3.metric("Forecast days", len(filtered))

    st.caption(
        f"Category: {selected_category} | "
        f"Department: {selected_department} | Model: {model}"
    )
    # Model evaluation using the selected product and store
    st.subheader("Model Evaluation (30-Day Backtest)")

    evaluation = backtest_df.copy()

    if not evaluation.empty:
        evaluation = evaluation[
            (evaluation["item_id"] == selected_item)
            & (evaluation["store_id"] == selected_store)
        ].copy()

        evaluation["actual_demand"] = pd.to_numeric(
            evaluation["actual_demand"], errors="coerce"
        )
        evaluation["predicted_demand"] = pd.to_numeric(
            evaluation["predicted_demand"], errors="coerce"
        )

        evaluation = evaluation.dropna(
            subset=["actual_demand", "predicted_demand"]
        )

    if not evaluation.empty:
        errors = (
            evaluation["actual_demand"]
            - evaluation["predicted_demand"]
        )

        mae = errors.abs().mean()
        rmse = (errors.pow(2).mean()) ** 0.5

        eval_col1, eval_col2, eval_col3 = st.columns(3)
        eval_col1.metric("MAE", f"{mae:.3f}")
        eval_col2.metric("RMSE", f"{rmse:.3f}")
        eval_col3.metric("Backtest days", len(evaluation))

        st.caption(
            "Metrics are calculated from held-out backtest actuals "
            "and predictions for the selected product and store. "
            "Lower MAE and RMSE generally indicate better accuracy."
        )
    else:
        st.warning(
            "No valid backtest rows are available for this product and store."
        )
    # What-if analysis
    st.subheader("What-if Analysis: Price Reduction")

    price_drop = st.slider(
        "Price reduction (%)",
        min_value=0,
        max_value=20,
        value=10,
        step=5
    )

    elasticity = st.slider(
        "Assumed price elasticity of demand",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help=(
            "Illustrative assumption, not learned from the current "
            "forecasting models. Higher values mean demand responds "
            "more strongly to price changes."
        )
    )

    scenario_df = filtered.copy()

    # Simplified constant-elasticity scenario.
    # This does not rerun Prophet or LightGBM.
    price_factor = 1 - price_drop / 100

    scenario_df["scenario_demand"] = (
        scenario_df["predicted_demand"]
        * (price_factor ** (-elasticity))
    )

    baseline_total = scenario_df["predicted_demand"].sum()
    scenario_total = scenario_df["scenario_demand"].sum()
    
    change_pct = (
        ((scenario_total / baseline_total) - 1) * 100
        if baseline_total > 0 else 0
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Baseline demand (forecast period)",
        f"{baseline_total:,.0f}"
    )

    metric2.metric(
        "Scenario demand (forecast period)",
        f"{scenario_total:,.0f}"
    )

    metric3.metric(
        "Illustrative demand change",
        f"{change_pct:+.1f}%"
    )

    st.caption(
        "Scenario estimates use an assumed elasticity. They are "
        "illustrative and are not validated model predictions."
    )

    st.line_chart(
        scenario_df.set_index("forecast_date")[
            ["predicted_demand", "scenario_demand"]
        ]
    )

    st.dataframe(
        scenario_df[
            [
                "forecast_date",
                "predicted_demand",
                "scenario_demand"
            ]
        ],
        use_container_width=True
    )

    # Forecast chart
    st.subheader("Forecasted Daily Demand")

    st.line_chart(
        filtered.set_index("forecast_date")["predicted_demand"]
    )

    # Complete forecast table
    st.subheader("Forecast Data")

    st.dataframe(
        filtered,
        use_container_width=True
    )

except Exception as e:
    st.error("Could not load forecast data from BigQuery.")
    st.exception(e)
    st.info(
        "Check Google Cloud authentication, table names, permissions, "
        "and the forecast table schemas."
    )
