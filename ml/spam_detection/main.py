import sys
from datetime import timedelta

import modal

image = modal.Image.debian_slim().pip_install(
    [
        "datasets~=2.7.1",
        "evaluate~=0.3.0",
        "loguru~=0.6.0",
        "scikit-learn~=1.1.3",  # Required by evaluate pkg.
        "torch~=1.13.0",
        "transformers~=4.24.0",
    ]
)
stub = modal.Stub(name="example-spam-detect-llm", image=image)


def _get_logger():
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, colorize=True)
    return logger


@stub.function(timeout=int(timedelta(minutes=30).total_seconds()))
def train():
    import numpy as np
    import evaluate
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification
    from transformers import AutoTokenizer
    from transformers import TrainingArguments, Trainer

    logger = _get_logger()

    logger.opt(colors=True).info(
        "Ready to detect <fg #9dc100><b>SPAM</b></fg #9dc100> from <fg #ffb6c1><b>HAM</b></fg #ffb6c1>?"
    )

    dataset = load_dataset("yelp_review_full")
    dataset["train"][100]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

    training_args = TrainingArguments(output_dir="test_trainer")

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    logger.opt(colors=True).info("<light-yellow>training</light-yellow> 🏋️")

    trainer.train()


if __name__ == "__main__":
    with stub.run():
        train()
