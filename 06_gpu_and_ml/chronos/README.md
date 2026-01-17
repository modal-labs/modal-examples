# Chronos-2 Time Series Forecasting on Modal

Deploy Amazon's [Chronos-2](https://huggingface.co/amazon/chronos-2) time series forecasting model on Modal.

These examples parallel the official [SageMaker deployment notebook](https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/deploy-chronos-to-amazon-sagemaker.ipynb), demonstrating equivalent functionality with Modal's simpler Python-native approach.

## Examples

- `real_time_inference.py` - Real-time forecasting with four patterns: univariate, multiple series, covariates, and multivariate
- `batch_transform.py` - Parallel batch processing using `.starmap()` for large-scale forecasting

## Quick Start

```bash
modal run 06_gpu_and_ml/chronos/real_time_inference.py
modal run 06_gpu_and_ml/chronos/batch_transform.py
```

## References

- [Chronos-2 on HuggingFace](https://huggingface.co/amazon/chronos-2)
- [SageMaker Deployment Notebook](https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/deploy-chronos-to-amazon-sagemaker.ipynb)
