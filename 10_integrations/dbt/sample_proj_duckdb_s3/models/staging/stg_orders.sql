with source as (

    {#-
    Here we load from the external S3 bucket data, which was seeded
    by running the `seed` Modal function.
    #}
    select * from {{ source('external_source', 'raw_orders') }}

),

renamed as (

    select
        id as order_id,
        user_id as customer_id,
        order_date,
        status

    from source

)

select * from renamed
