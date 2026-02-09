# [Chronos-2](https://huggingface.co/amazon/chronos-2) is a 120M-parameter encoder-only time series foundation model for zero-shot forecasting of univariate and multivariate time series; it also supports covariates.

# Chronos-2 can be deployed to AWS with SageMaker by following the official Amazon Science [guide](https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/deploy-chronos-to-amazon-sagemaker.ipynb); it uses ~100 lines of helper code to convert a pandas data frame to a JSON payload and back because SageMaker endpoints communicate over HTTP.

# Deployment to Modal is simpler.

# Modal uses python-native RPC (Remote Procedure Call) and data frames pass between the local machine and the remote container without explicit serialization to JSON. Behind the scenes, Modal handles the serialization automatically using cloudpickle.

# Here's how to run two patterns from the Chronos-2 SageMaker tutorial on the Modal cloud:
# * Real-time inference for univariate, multivariate and covariate-informed forecasting.
# * Batch transform to process thousands of time series in parallel.


import modal
import pandas as pd

app = modal.App("chronos-forecasting")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "chronos-forecasting>=2.0",
    "pandas>=2.3",
    "pyarrow",
)


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
    def predict_df(self, *args, **kwargs) -> pd.DataFrame:
        return self.pipeline.predict_df(*args, **kwargs)


# ## Real-time Inference
#
# Chronos-2 can forecast univariate and multivariate time series, with and without covariates.

# We build the input data frame: it requires an item identifier, a timestamp, one or more targets to forecast and optional covariates. When we call `predict_df.remote()`, Modal serializes the data frame via cloudpickle to send to the cloud container; once the forecast is generated, it uses cloudpickle again to serialize the result and return it as a data frame.


def real_time_inference(model):
    print("(a) Univariate forecasting")
    context_df = pd.DataFrame(
        {
            "item_id": "id",
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="D"),
            "target": [
                0.0,
                4.0,
                5.0,
                1.5,
                -3.0,
                -5.0,
                -3.0,
                1.5,
                5.0,
                4.0,
                0.0,
                -4.0,
                -5.0,
                -1.5,
                3.0,
                5.0,
                3.0,
                -1.5,
                -5.0,
                -4.0,
            ],
        }
    )
    print(model.predict_df.remote(context_df, prediction_length=10))

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
    print(model.predict_df.remote(context_df, prediction_length=5))

    print("(c) Forecasting with covariates")
    context_df = pd.DataFrame(
        {
            "item_id": "id",
            "timestamp": pd.date_range("2026-01-01", periods=9, freq="D"),
            "target": [1.0, 2.0, 3.0, 2.0, 0.5, 2.0, 3.0, 2.0, 1.0],
            "feat_1": [3.0, 6.0, 9.0, 6.0, 1.5, 6.0, 9.0, 6.0, 3.0],
            "feat_2": ["A", "B", "B", "B", "A", "A", "A", "A", "B"],
            "feat_3": [10, 20, 30, 20, 5, 20, 30, 20, 10],  # past only
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
    print(model.predict_df.remote(context_df, future_df=future_df, prediction_length=3))

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
    print(
        model.predict_df.remote(
            context_df, target=["target_1", "target_2", "target_3"], prediction_length=4
        )
    )


# ## Batch Transform
#
# In this example, we forecast multiple time series (with covariates) in parallel. First we batch the training data `past_df` and the future covariates `future_df`. Then we fan out the `predict_df()` call using `.starmap()`.


def batch_transform(model):
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
            past_df[past_df["item_id"].isin(series_ids[i : i + batch_size])],
            future_df[future_df["item_id"].isin(series_ids[i : i + batch_size])],
        )
        for i in range(0, len(series_ids), batch_size)
    ]

    # Process batches in parallel with .starmap()
    print(f"Processing {len(series_ids)} series in {len(batches)} batches...")
    predictions = pd.concat(
        model.predict_df.starmap(
            batches,
            kwargs={"prediction_length": prediction_length, "target": target_col},
        )
    )
    print(predictions.head(15))


# ## Run the examples
#
# ```bash
# modal run misc/chronos_forecasting.py
# ```


@app.local_entrypoint()
def main():
    model = Chronos()

    real_time_inference(model)
    batch_transform(model)
