
# Retail Demand Forecasting & Inventory Optimization

An end-to-end retail analytics project that forecasts product demand using historical Walmart sales data. The project combines Python, SQL, Google BigQuery, dbt, Prophet, LightGBM, and Streamlit to support data-driven inventory planning.

## Project Objectives

- Analyze historical retail sales data.
- Perform data quality checks and data transformation.
- Build demand forecasting models using Prophet and LightGBM.
- Evaluate forecast accuracy using MAE and RMSE.
- Visualize future demand through an interactive dashboard.
- Explore illustrative price-reduction scenarios to understand potential changes in demand.

## Tech Stack

- **Programming:** Python, SQL
- **Data Warehouse:** Google BigQuery
- **Data Transformation:** dbt
- **Forecasting:** Prophet, LightGBM
- **Data Processing:** Pandas, NumPy
- **Dashboard:** Streamlit
- **Version Control:** Git, GitHub

## Dataset

This project uses the M5 Forecasting dataset based on Walmart retail sales.

The dataset includes:

- `sales_train_validation.csv` — historical sales data
- `sales_train_evaluation.csv` — evaluation sales data
- `calendar.csv` — calendar dates and event information
- `sell_prices.csv` — item prices by store and week

The original dataset is large and should be downloaded separately rather than committed to GitHub.

## Project Workflow

### 1. Data Ingestion and Quality Checks

- Loaded the M5 dataset into BigQuery.
- Organized raw tables in the `m5_raw` dataset.
- Checked row counts, duplicate identifiers, missing values, and invalid sales and prices.
- Validated calendar date continuity.

### 2. Data Transformation with dbt

Created staging and intermediate models to prepare analytics-ready data.

Models include:

- `stg_calendar.sql`
- `stg_sales.sql`
- `stg_sell_prices.sql`
- `int_daily_sales.sql`
- `int_weekly_sales.sql`
- `int_monthly_sales.sql`
- `forecast_base.sql`

Transformed data is stored in the `m5_analytics` dataset.

### 3. Demand Forecasting

**Prophet**
- Models demand trends and seasonality.
- Uses calendar event information.
- Generates 30-day future forecasts.

**LightGBM**
- Applies machine learning to demand forecasting.
- Produces product-store-level forecasts.

Both models use a 30-day backtest to evaluate prediction accuracy.

### 4. Model Evaluation

Forecast performance is evaluated using:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted demand.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more heavily.

Lower MAE and RMSE generally indicate better forecast accuracy. The dashboard displays these metrics for the selected product and store.

### 5. Interactive Dashboard

The Streamlit dashboard provides:

- Prophet and LightGBM model selection
- Category, department, product, and store filters
- Daily demand forecast charts
- Forecast data tables
- MAE and RMSE evaluation metrics
- Price-reduction what-if analysis

**Note:** The price-reduction scenario uses an assumed price elasticity. It is an illustrative calculation and does not retrain the forecasting models or establish a causal relationship between price and demand.

## Project Structure

```text
retail_demand_forecast/
├── data/
│   └── raw/
├── ingestion/
├── sql/
├── models/
│   ├── staging/
│   ├── intermediate/
│   └── forecasting/
├── forecasting/
│   ├── prophet_forecast.py
│   └── lightgbm_forecast.py
├── dashboard.py
├── dbt_project.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup and Execution

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd retail_demand_forecast
```

Replace the placeholder with your actual repository URL.

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Google Cloud authentication

```bash
gcloud auth application-default login
```

Ensure BigQuery is enabled and your account has the required permissions.

### 5. Prepare the data and transformations

Download the M5 dataset, configure the BigQuery tables, run the ingestion scripts, and execute the dbt models according to the project configuration.

### 6. Generate forecasts

Run the scripts from the project root:

```bash
python forecasting/prophet_forecast.py
python forecasting/lightgbm_forecast.py
```

### 7. Launch the dashboard

```bash
streamlit run dashboard.py
```

Open the local URL displayed in the terminal.

## Future Improvements

- Expand forecasts to more products and stores.
- Compare model performance across multiple product-store combinations.
- Incorporate inventory levels and supplier lead times.
- Develop automated restocking recommendations.
- Schedule forecast refreshes and monitor model performance.

## Author

**Abhijith P. Anil**

B.Tech Computer Science Engineering | Aspiring Data Analyst

GitHub: [AbhijithAJ3](https://github.com/AbhijithAJ3)
