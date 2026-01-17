# ---
# cmd: ["modal", "run", "06_gpu_and_ml/chronos/real_time_inference.py"]
# ---

# # Deploy Chronos-2 to Modal

# This tutorial shows how to deploy Chronos-2 to [Modal](https://modal.com), following the structure of the [official SageMaker deployment notebook](https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/deploy-chronos-to-amazon-sagemaker.ipynb).

# ### Why Deploy to Modal?

# Running models locally works for experimentation, but production use cases need reliability, scale, and integration into existing workflows. Modal lets you deploy Chronos-2 to the cloud with:

# - Fast cold starts (~25 seconds including model loading, compared to ~5 minutes for SageMaker endpoint provisioning)
# - Per-second billing (pay only for compute time)
# - Automatic scaling (no instance management)
# - No infrastructure setup (no IAM roles, endpoint configs, or S3 model artifacts)
# - Native Python RPC (no JSON conversion overhead)

# RPC stands for Remote Procedure Call—it lets you call functions running on Modal's servers as if they were local Python functions. Combined with Parquet serialization, this avoids the complex JSON conversion code required by HTTP-based endpoints.

# ### Deployment Options

# This tutorial covers two deployment patterns from the SageMaker notebook:

# 1. Real-time Inference (GPU) - Highest throughput, ~25 second cold starts
# 2. Batch Transform - Process thousands of time series in parallel (see batch_transform.py)

# Unlike SageMaker, all Modal deployments scale to zero automatically and bill per-second. There's no separate "serverless" mode because Modal is serverless by default.
import io

import modal
import pandas as pd

app = modal.App("chronos-forecasting")


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


@app.local_entrypoint()
def main():
    # Demonstrates four forecasting patterns from the SageMaker notebook.
    model = Chronos()

    print("Section 1: Real-time Inference (GPU)")

    print("(a) Univariate forecasting")
    context_df = pd.DataFrame(
        {
            "item_id": "id",
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="D"),
            "target": [0, 4, 5, 1.5, -3, -5, -3, 1.5, 5, 4, 0, -4, -5, -1.5, 3, 5, 3, -1.5, -5, -4],
        }
    )
    pred_bytes = model.predict.remote(df_to_bytes(context_df), prediction_length=10)
    print(bytes_to_df(pred_bytes))

    print("(b) Multiple time series")
    product_a = pd.DataFrame(
        {
            "item_id": "product_A",
            "timestamp": pd.date_range("2024-01-01T01:00:00", periods=9, freq="1h"),
            "target": [1.0, 2.0, 3.0, 2.0, 0.5, 2.0, 3.0, 2.0, 1.0],
        }
    )
    product_b = pd.DataFrame(
        {
            "item_id": "product_B",
            "timestamp": pd.date_range("2024-02-02T03:00:00", periods=7, freq="1h"),
            "target": [5.4, 3.0, 3.0, 2.0, 1.5, 2.0, -1.0],
        }
    )
    context_df = pd.concat([product_a, product_b], ignore_index=True)
    pred_bytes = model.predict.remote(df_to_bytes(context_df), prediction_length=5)
    print(bytes_to_df(pred_bytes))

    print("(c) Forecasting with covariates")
    context_df = pd.DataFrame(
        {
            "item_id": "id",
            "timestamp": pd.date_range("2026-01-01", periods=9, freq="D"),
            "target": [1.0, 2.0, 3.0, 2.0, 0.5, 2.0, 3.0, 2.0, 1.0],
            "feat_1": [3.0, 6.0, 9.0, 6.0, 1.5, 6.0, 9.0, 6.0, 3.0],
            "feat_2": ["A", "B", "B", "B", "A", "A", "A", "A", "B"],
            "feat_3": [10.0, 20.0, 30.0, 20.0, 5.0, 20.0, 30.0, 20.0, 10.0],  # past only
        }
    )
    future_df = pd.DataFrame(
        {
            "item_id": "id",
            "timestamp": pd.date_range("2026-01-10", periods=3, freq="D"),
            "feat_1": [2.5, 2.2, 3.3],
            "feat_2": ["B", "A", "A"],
        }
    )
    pred_bytes = model.predict.remote(
        df_to_bytes(context_df),
        future_df_bytes=df_to_bytes(future_df),
        prediction_length=3,
    )
    print(bytes_to_df(pred_bytes))

    print("(d) Multivariate forecasting")
    context_df = pd.DataFrame(
        {
            "item_id": "id",
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="D"),
            "target_1": [1, 2, 3, 2, 1, 2, 3, 4.0],
            "target_2": [5, 4, 3, 4, 5, 4, 3, 2.0],
            "target_3": [2, 2.5, 3, 2.5, 2, 2.5, 3, 3.5],
        }
    )
    pred_bytes = model.predict.remote(
        df_to_bytes(context_df),
        target=["target_1", "target_2", "target_3"],
        prediction_length=4,
    )
    print(bytes_to_df(pred_bytes))
