## Spam detection with large-language models (LLMs)

This demo application tackles the internet's old bête noire: spam. To focus of the demo
is fine-tuning the BERT LLM to acheive XX% on an email spam classification task, but any
self-respecting LLM should be able to beat a baseline model, so this demo also includes
other model architectures for comparision.

The models are:

- _Bidirectional Encoder Representations from Transformers_ (BERT)
- TODO: Some older RNN
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

TODO

### Training

TODO

```bash
cd "$(git rev-parse --show-toplevel)/ml/spam_detect"
echo TODO
```

Sending <kbd>Ctrl</kbd>+<kbd>C</kbd> will stop your app.

### Serving

## Deploy

To deploy this application's spam detection API, run the following command:

```bash
cd "$(git rev-parse --show-toplevel)/ml/spam_detect"
echo TODO
```
