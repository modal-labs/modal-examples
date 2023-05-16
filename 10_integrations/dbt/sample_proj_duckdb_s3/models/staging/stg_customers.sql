with source as (

    {#-
    Here we load from the external S3 bucket data, which was seeded
    by running the `seed` Modal function.
    #}
    select * from {{ source('external_source', 'raw_customers') }}

),

renamed as (

    select
        id as customer_id,
        first_name,
        last_name

    from source

)

select * from renamed
