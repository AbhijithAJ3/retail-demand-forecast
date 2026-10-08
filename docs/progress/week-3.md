# Week 3 Progress — Retail Demand Forecasting & Inventory Optimization

## Completed Work

### 1. Prophet Forecasting
- Implemented Prophet-based demand forecasting.
- Used yearly and weekly seasonality.
- Incorporated M5 calendar events as holidays.
- Selected three high-volume item/store series for forecasting.
- Performed a 30-day backtest using MAE and RMSE.
- Generated 30-day future forecasts.
- Stored Prophet forecast outputs in BigQuery.

### 2. LightGBM Forecasting
- Implemented LightGBM for multi-variable demand forecasting.
- Created time-series features:
  - Lag 1
  - Lag 7
  - Lag 14
  - Lag 28
  - 7-day rolling mean
  - 28-day rolling mean
- Used calendar and pricing features:
  - Sell price
  - Year
  - Month
  - Day of week
  - Week of year
- Used a time-based 30-day recursive backtest.
- Evaluated the model using MAE and RMSE.
- Generated 30-day recursive future forecasts.

### 3. LightGBM Evaluation

| Item | Store | MAE | RMSE |
|------|-------|-----|------|
| FOODS_3_090 | CA_3 | 20.29 | 28.69 |
| FOODS_3_586 | TX_2 | 14.46 | 17.35 |
| FOODS_3_586 | TX_3 | 15.35 | 17.45 |

### 4. BigQuery Forecast Output

LightGBM outputs were stored in:

`m5_analytics.lightgbm_forecast_outputs`

Output:
- 90 backtest rows
- 90 future forecast rows
- 180 total rows

### Week 3 Status

- Prophet forecasting: Complete
- LightGBM forecasting: Complete
- Forecast evaluation: Complete
- Forecast outputs stored in BigQuery: Complete

## Next Step

Proceed to Week 4 dashboard development using Streamlit, including forecast visualization and what-if analysis.