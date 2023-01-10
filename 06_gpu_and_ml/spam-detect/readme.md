## Spam detection with large-language models (LLMs)

This demo application tackles the internet's old bête noire: spam. To focus of the demo
is fine-tuning the BERT LLM to acheive 99% precision on an email spam classification task, but
any self-respecting LLM should be able to beat a baseline model, so this demo also includes
other model architectures for comparision.

The models are:

- _Bidirectional Encoder Representations from Transformers_ (BERT)
- _Naive Bayes_
- Blacklist-based hueristic model (`BadWords`)

### ML engineering support

The demo application showcases how to do all the major machine learning engineering processes:

- training dataset preparation
- model training and evaluation loop
- model tracking and storage
- model deployment (promotion, rollback)
- model serving

## Developing

> Run all these commands in the project's directory: `cd "$(git rev-parse --show-toplevel)/ml/spam_detect"`.
### Testing

Unit-tests exist in `tests/` and can be run with:

```
python3 -m pytest
```
### Training

Any of the models can be trained using the `train.py` module.

```bash
python3 -m spam_detect.train
```
### Serving

```
cd "$(git rev-parse --show-toplevel)/06_gpu_and_ml/spam-detect"
python3 -m spam_detect.serving
```

Sending <kbd>Ctrl</kbd>+<kbd>C</kbd> will stop your app serving.

## Deploy

To deploy this application's spam detection API, run the following command:

```bash
cd "$(git rev-parse --show-toplevel)/06_gpu_and_ml/spam-detect"
modal deploy spam_detect.serving
```

### Deploying models

TODO
