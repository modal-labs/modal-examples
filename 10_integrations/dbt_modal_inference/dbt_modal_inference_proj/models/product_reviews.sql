with products as (

    select * from {{ ref('stg_products') }}

),

reviews as (

    select * from {{ ref('stg_reviews') }}

),

product_reviews as (

    select 
        p.id as product_id,
        p.name as product_name,
        p.description as product_description,
        r.review as product_review
    from products p
    left join reviews r on p.id = r.product_id
)

select * from product_reviews
