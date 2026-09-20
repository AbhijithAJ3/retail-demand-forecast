SELECT
    id,
    item_id,
    dept_id,
    cat_id,
    store_id,
    state_id,
    d,
    sales
FROM {{ source('m5_raw', 'sales_train_validation') }}
UNPIVOT (
    sales FOR d IN (
        {% for i in range(1, 1914) %}
            d_{{ i }}{% if not loop.last %},{% endif %}
        {% endfor %}
    )
)