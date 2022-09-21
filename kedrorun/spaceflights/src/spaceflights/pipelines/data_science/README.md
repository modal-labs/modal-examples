# Data Science pipeline

> *Note:* This `README.md` was generated using `Kedro 0.18.2` for illustration purposes. Please modify it according to your pipeline structure and contents.

## Overview

This modular pipeline:
1. Splits the model input table into train and test subsets (`split_data_node`)
2. Trains a simple linear regression model (`train_model_node`)
3. Evaluates the performance of the trained model on the test set (`evaluate_model_node`)

## Pipeline inputs

### `model_input_table`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | A combined dataset containing data on shuttles, joined with company and reviews information |

### `parameters`

|      |                    |
| ---- | ------------------ |
| Type | `dict` |
| Description | Project parameter dictionary that must contain the following keys: `test_size` (the proportion of the dataset to include in the test split), `random_state` (random seed for the shuffling applied to the data before applying the split), `features` (list of features to use for modelling) |


## Pipeline outputs

### `X_train`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | Train set features |

### `y_train`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.Series` |
| Description | Train set target variable |

### `X_test`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | Test set features |

### `y_test`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.Series` |
| Description | Test set target variable |

### `regressor`

|      |                    |
| ---- | ------------------ |
| Type | `sklearn.linear_model.LinearRegression` |
| Description | Trained linear regression model |
