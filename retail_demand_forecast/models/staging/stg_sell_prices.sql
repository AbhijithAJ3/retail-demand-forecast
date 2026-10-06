SELECT
    store_id,
    item_id,
    wm_yr_wk,
    sell_price
FROM {{ source('m5_raw', 'sell_prices') }}