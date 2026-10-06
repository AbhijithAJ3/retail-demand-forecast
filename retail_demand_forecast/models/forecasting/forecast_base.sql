SELECT
    item_id,
    store_id,
    dept_id,
    cat_id,
    state_id,
    date,
    sales,
    sell_price,

    EXTRACT(YEAR FROM date) AS year,
    EXTRACT(MONTH FROM date) AS month,
    EXTRACT(DAYOFWEEK FROM date) AS day_of_week,
    EXTRACT(WEEK FROM date) AS week_of_year

FROM {{ ref('int_daily_sales') }}