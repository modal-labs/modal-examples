with source as (

    select * from {{ source('external_source', 'raw_products') }}

)

select * from source
