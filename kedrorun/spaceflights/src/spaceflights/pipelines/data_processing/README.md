# Data Processing pipeline
## Overview

This modular pipeline preprocesses the raw data (`preprocess_companies_node` and `preprocess_shuttles_node`) and creates the model input table (`create_model_input_table_node`).

## Pipeline inputs

### `companies`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | Raw data on the companies running the space shuttles |

### `shuttles`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | Raw data on technical characteristics of the space shuttles |

### `reviews`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | Raw data with historical customer reviews of their space trips |


## Pipeline outputs

### `preprocessed_companies`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | Preprocessed version of the `companies` dataset |

### `preprocessed_shuttles`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | Preprocessed version of the `shuttles` dataset |

### `model_input_table`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | A combined dataset containing data on shuttles, joined with company and reviews information |
