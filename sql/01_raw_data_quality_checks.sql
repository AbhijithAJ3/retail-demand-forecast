-- ============================================================
-- M5 Retail Demand Forecasting
-- Raw Data Quality Checks
-- ============================================================

-- 1. Row counts
SELECT
  'calendar' AS table_name,
  COUNT(*) AS row_count
FROM `gen-lang-client-0915147620.m5_raw.calendar`

UNION ALL

SELECT
  'sales_train_validation',
  COUNT(*)
FROM `gen-lang-client-0915147620.m5_raw.sales_train_validation`

UNION ALL

SELECT
  'sales_train_evaluation',
  COUNT(*)
FROM `gen-lang-client-0915147620.m5_raw.sales_train_evaluation`

UNION ALL

SELECT
  'sell_prices',
  COUNT(*)
FROM `gen-lang-client-0915147620.m5_raw.sell_prices`;


-- 2. Duplicate ID validation
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT id) AS unique_ids,
  COUNT(*) - COUNT(DISTINCT id) AS duplicate_rows
FROM `gen-lang-client-0915147620.m5_raw.sales_train_validation`;


-- 3. Duplicate price-key validation
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT CONCAT(store_id, '|', item_id, '|', CAST(wm_yr_wk AS STRING)))
    AS unique_price_keys,
  COUNT(*) -
  COUNT(DISTINCT CONCAT(store_id, '|', item_id, '|', CAST(wm_yr_wk AS STRING)))
    AS duplicate_rows
FROM `gen-lang-client-0915147620.m5_raw.sell_prices`;


-- 4. Sales volume validation
SELECT
  MIN(sales_volume) AS minimum_sales,
  MAX(sales_volume) AS maximum_sales,
  COUNTIF(sales_volume < 0) AS negative_sales_values,
  COUNTIF(sales_volume IS NULL) AS null_sales_values
FROM `gen-lang-client-0915147620.m5_raw.sales_train_validation`
UNPIVOT (
  sales_volume FOR sales_day IN (
    d_1, d_2, d_3, d_4, d_5, d_6, d_7
  )
);


-- 5. Sell price validation
SELECT
  MIN(sell_price) AS minimum_price,
  MAX(sell_price) AS maximum_price,
  COUNTIF(sell_price < 0) AS negative_prices,
  COUNTIF(sell_price IS NULL) AS null_prices
FROM `gen-lang-client-0915147620.m5_raw.sell_prices`;


-- 6. Calendar date range
SELECT
  MIN(date) AS earliest_date,
  MAX(date) AS latest_date,
  COUNT(*) AS total_calendar_days
FROM `gen-lang-client-0915147620.m5_raw.calendar`;


-- 7. Calendar continuity
SELECT
  COUNT(*) AS calendar_days,
  DATE_DIFF(MAX(date), MIN(date), DAY) + 1 AS expected_days,
  (DATE_DIFF(MAX(date), MIN(date), DAY) + 1) - COUNT(*) AS missing_days
FROM `gen-lang-client-0915147620.m5_raw.calendar`;


-- 8. Sales date-column validation
SELECT
  COUNT(*) AS total_columns,
  COUNTIF(STARTS_WITH(column_name, 'd_')) AS sales_day_columns,
  MIN(SAFE_CAST(SUBSTR(column_name, 3) AS INT64)) AS first_sales_day,
  MAX(SAFE_CAST(SUBSTR(column_name, 3) AS INT64)) AS last_sales_day
FROM `gen-lang-client-0915147620.m5_raw.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'sales_train_validation';