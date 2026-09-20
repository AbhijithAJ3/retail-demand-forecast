SELECT
    date,
    wm_yr_wk,
    weekday,
    wday,
    month,
    year,
    event_name_1,
    event_type_1,
    event_name_2,
    event_type_2,
    snap_CA,
    snap_TX,
    snap_WI,
    d
FROM {{ source('m5_raw', 'calendar') }}