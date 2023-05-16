with source as (
    
    {#-
    Here we load from the external S3 bucket data, which was seeded
    by running the `seed` Modal function.
    #}
    select * from {{ source('external_source', 'raw_payments') }}

),

renamed as (

    select
        id as payment_id,
        order_id,
        payment_method,

        -- `amount` is currently stored in cents, so we convert it to dollars
        amount / 100 as amount

    from source

)

select * from renamed
