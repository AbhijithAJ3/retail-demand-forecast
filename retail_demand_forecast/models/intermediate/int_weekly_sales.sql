SELECT
    item_id,
    dept_id,
    cat_id,
    store_id,
    state_id,

    wm_yr_wk,

    MIN(date) AS week_start_date,
    MAX(date) AS week_end_date,

    SUM(sales) AS total_sales,

    AVG(sell_price) AS avg_sell_price,

    COUNT(DISTINCT date) AS days_with_data

FROM {{ ref('int_daily_sales') }}

GROUP BY
    item_id,
    dept_id,
    cat_id,
    store_id,
    state_id,
    wm_yr_wk