with source as (

    select * from {{ source('external_source', 'raw_reviews') }}

)

select * from source
