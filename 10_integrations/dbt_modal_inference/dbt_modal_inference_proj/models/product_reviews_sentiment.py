import json

import pyarrow as pa
import requests


def get_prompt(review):
    """
    This function takes a review and returns a prompt for the review sentiment classification.

    Args:
        review: A product review.

    Returns:
        A prompt for the review sentiment classification.
    """
    return (
        """
You are an expert at analyzing product reviews sentiment.
Your task is to classify the given product review into one of the following labels: ["positive", "negative", "neutral"]
Here are some examples:
1. "example": "Packed with innovative features and reliable performance, this product exceeds expectations, making it a worthwhile investment."
   "label": "positive"
2. "example": "Despite promising features, the product's build quality and performance were disappointing, failing to meet expectations."
   "label": "negative"
3. "example": "While the product offers some useful functionalities, its overall usability and durability may vary depending on individual needs and preferences."
   "label": "neutral"
Label the following review:
"""
        + '"'
        + review
        + '"'
        + """
Respond in a single word with the label.
"""
    )


def batcher(batch_reader: pa.RecordBatchReader, inference_url: str):
    """
    This function takes a batch reader and an inference url and yields a record batch with the review sentiment.

    Args:
        batch_reader: A record batch reader.
        inference_url: The url of the inference service.

    Yields:
        A record batch with the review sentiment.
    """
    for batch in batch_reader:
        df = batch.to_pandas()

        prompts = (
            df["product_review"]
            .apply(lambda review: get_prompt(review))
            .tolist()
        )

        res = (
            requests.post(  # request to the inference service running on Modal
                inference_url,
                json={"prompts": prompts},
            )
        )

        df["review_sentiment"] = json.loads(res.content)

        yield pa.RecordBatch.from_pandas(df)


def model(dbt, session):
    """
    This function defines the model for the product reviews sentiment.

    Args:
        dbt: The dbt object.
        session: The session object.

    Returns:
        A record batch reader with the review sentiment.
    """
    dbt.config(
        materialized="external",
        location="/root/vol/db/review_sentiments.parquet",
    )
    inference_url = dbt.config.get("inference_url")

    big_model = dbt.ref("product_reviews")
    batch_reader = big_model.record_batch(100)
    batch_iter = batcher(batch_reader, inference_url)
    new_schema = batch_reader.schema.append(
        pa.field("review_sentiment", pa.string())
    )
    return pa.RecordBatchReader.from_batches(new_schema, batch_iter)
