# Retail Demand Forecasting & Inventory Optimization

An end-to-end retail analytics platform designed to analyze historical sales, pricing, calendar events, and seasonality to forecast future product demand and support data-driven inventory planning.

The project uses the **M5 Forecasting Dataset**, which contains historical Walmart sales data across products, departments, stores, and dates.

---

## 🎯 Project Objective

The goal of this project is to build a data pipeline and forecasting system that can help retail teams:

- Forecast future product demand
- Identify demand patterns and seasonality
- Analyze sales at store and product levels
- Support proactive inventory planning
- Reduce the risk of stockouts and overstocking
- Improve supply-chain decision making

---

## 🗂️ Dataset

### M5 Forecasting Dataset

Source: Walmart historical sales data from the M5 Forecasting competition.

The dataset contains:

- Product information
- Department and category information
- Store and state information
- Daily historical sales
- Weekly selling prices
- Calendar and event information

### Main Raw Tables

| Table | Description |
|---|---|
| `calendar` | Dates, weekdays, months, years, events and SNAP indicators |
| `sales_train_validation` | Historical daily sales for products and stores |
| `sales_train_evaluation` | Extended sales data used for evaluation |
| `sell_prices` | Weekly selling prices by product and store |

---

## 🏗️ Architecture

```text
                    M5 Forecasting Dataset
                              │
                              ▼
                    Local CSV Data Files
                              │
                              ▼
                     Google BigQuery
                         m5_raw
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
          calendar      sales_train      sell_prices
                         validation
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                         dbt Sources
                              │
                              ▼
                       Staging Models
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       stg_calendar       stg_sales      stg_sell_prices
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    Intermediate Models
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
          int_daily_sales  int_weekly   int_monthly
                           _sales        _sales
                              │
                              ▼
                     Forecasting Layer
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                 Prophet             LightGBM
                    │                   │
                    └─────────┬─────────┘
                              ▼
                     Forecast Outputs
                              │
                              ▼
                     Streamlit Dashboard