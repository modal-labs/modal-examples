# ---
# cmd: ["modal", "run", "06_gpu_and_ml/chronos/batch_transform.py"]
# ---

# # Batch Transform for Time Series
#
# Process thousands of time series in parallel using Modal's `.starmap()` for
# horizontal scaling. This is Modal's equivalent of SageMaker Batch Transform.
#
# **When to use:**
# - Large-scale batch forecasting (thousands of time series)
# - Scheduled or periodic forecasting jobs
# - When latency is not critical

import io

import modal
import pandas as pd

app = modal.App("chronos-batch-transform")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "chronos-forecasting>=2.0",
    "pandas[pyarrow]",
)


def df_to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf)
    return buf.getvalue()


def bytes_to_df(b: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(b))


# ## Deploy the Model


@app.cls(image=image, gpu="T4")
class Chronos:
    @modal.enter()
    def load_model(self):
        from chronos import Chronos2Pipeline

        self.pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            device_map="cuda",
        )

    @modal.method()
    def predict(
        self,
        df_bytes: bytes,
        future_df_bytes: bytes | None = None,
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        target: str | list[str] = "target",
        prediction_length: int | None = None,
        quantile_levels: list[float] = [0.1, 0.5, 0.9],
        batch_size: int = 256,
        context_length: int | None = None,
        cross_learning: bool = False,
        validate_inputs: bool = True,
        **predict_kwargs,
    ) -> bytes:
        # Thin wrapper around pipeline.predict_df with Parquet serialization for RPC.
        pred_df = self.pipeline.predict_df(
            bytes_to_df(df_bytes),
            future_df=bytes_to_df(future_df_bytes) if future_df_bytes else None,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target=target,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            batch_size=batch_size,
            context_length=context_length,
            cross_learning=cross_learning,
            validate_inputs=validate_inputs,
            **predict_kwargs,
        )
        return df_to_bytes(pred_df)


# ## Run Batch Transform
#
# Uses the [grocery_sales dataset](https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/test.csv)
# from the official Chronos SageMaker tutorial.


@app.local_entrypoint()
def main():
    # Load grocery sales dataset (same as SageMaker tutorial)
    df = pd.read_csv(
        "https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/test.csv",
        parse_dates=["timestamp"],
    )

    prediction_length = 8
    target_col = "unit_sales"

    # Use historical data as context (exclude last prediction_length rows per series)
    past_df = df.groupby("item_id").head(-prediction_length)
    future_df = df.groupby("item_id").tail(prediction_length).drop(columns=[target_col])

    # Split into batches of 100 series each
    series_ids = past_df["item_id"].unique()
    batch_size = 100
    batches = [
        (
            df_to_bytes(past_df[past_df["item_id"].isin(series_ids[i : i + batch_size])]),
            df_to_bytes(future_df[future_df["item_id"].isin(series_ids[i : i + batch_size])]),
        )
        for i in range(0, len(series_ids), batch_size)
    ]

    # Process batches in parallel with .starmap()
    print(f"Processing {len(series_ids)} series in {len(batches)} batches...")
    model = Chronos()
    results = list(
        model.predict.starmap(
            batches,
            kwargs={"prediction_length": prediction_length, "target": target_col},
        )
    )

    # Combine results
    all_predictions = pd.concat([bytes_to_df(pred_bytes) for pred_bytes in results])
    print(
        f"Generated {len(all_predictions)} predictions for {all_predictions['item_id'].nunique()} series"
    )
    print(all_predictions.head(15))
