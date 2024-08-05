with product_reviews_sentiment as (

    select 
        product_id,
        product_name,
        product_description,
        product_review,
        review_sentiment,
      from {{ ref('product_reviews_sentiment') }}
),

clean as (

    select 
        product_id,
        product_name,
        product_description,
        product_review,
        case when regexp_matches(review_sentiment, 'positive', 'i') then 'positive' else null end AS positive_reviews,
        case when regexp_matches(review_sentiment, 'neutral', 'i') then 'neutral' else null end AS neutral_reviews,
        case when regexp_matches(review_sentiment, 'negative', 'i') then 'negative' else null end AS negative_reviews
    from product_reviews_sentiment

),

aggregated as (

    select
        product_name,
        count(positive_reviews) as positive_reviews,
        count(neutral_reviews) as neutral_reviews,
        count(negative_reviews) as negative_reviews
    from clean
    group by 1
    order by 2 desc

)

select * from aggregated