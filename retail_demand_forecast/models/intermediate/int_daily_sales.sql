SELECT
    s.id,
    s.item_id,
    s.dept_id,
    s.cat_id,
    s.store_id,
    s.state_id,

    s.d AS sales_day,
    c.date,
    c.wm_yr_wk,
    c.weekday,
    c.wday,
    c.month,
    c.year,

    c.event_name_1,
    c.event_type_1,
    c.event_name_2,
    c.event_type_2,

    c.snap_CA,
    c.snap_TX,
    c.snap_WI,

    p.sell_price,

    s.sales

FROM {{ ref('stg_sales') }} AS s

LEFT JOIN {{ ref('stg_calendar') }} AS c
    ON s.d = c.d

LEFT JOIN {{ ref('stg_sell_prices') }} AS p
    ON s.store_id = p.store_id
    AND s.item_id = p.item_id
    AND c.wm_yr_wk = p.wm_yr_wk