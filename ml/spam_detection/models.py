import math
import pathlib
import re
from collections import defaultdict

from .datasets.enron import structure
from . import config
from . import model_trainer

from typing import (
    Callable,
    Iterable,
    Protocol,
)

Prediction = float
Dataset = Iterable[structure.Example]
SpamClassifier = Callable[[str], Prediction]


class SpamModel(Protocol):
    def train(self, dataset: Dataset) -> SpamClassifier:
        ...

    def load(self, model_path: pathlib.Path) -> SpamClassifier:
        ...

    def save(self, model_path: pathlib.Path) -> str:
        ...


def train_llm_classifier(dataset: Dataset):
    import numpy as np
    import evaluate
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification
    from transformers import AutoTokenizer
    from transformers import TrainingArguments, Trainer

    logger = config._get_logger()

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

    dest_path = config.MODEL_STORE_DIR / "tmpmodel"
    dest_path.mkdir(parents=True, exist_ok=True)
    logger.opt(colors=True).info(f"<light-green>✔️ training done!</light-green>: saving model to {dest_path}")
    trainer.save_model(output_dir=dest_path)


def bad_words_spam_classifier(email: str) -> Prediction:
    """
    An extremely rudimentary heuritistic model. If a trained model
    can't beat this something is very wrong.
    """
    tokens = " ".split(email)
    tokens_set = set(tokens)
    bad_words = {
        "sex",
        "xxx",
        "nigerian",
        "teens",
    }
    max_bad_words = 2
    bad_words_count = 0
    for word in bad_words:
        if word in tokens_set:
            bad_words_count += 1
    return 1.0 if bad_words_count > max_bad_words else 0.0


# TODO(Jonathon): Calculate spam email's N most popular non-stop-words to use as spam indicators.
# This is basically a smarter version of `bad_words_spam_classifier`, which is the dumbest classifier,
# using a fixed set of words and ignoring the available dataset.
def build_top_spam_words_classifier(ds: Dataset) -> SpamClassifier:
    def classifier(email: str) -> Prediction:
        return 0.0

    return classifier


def tokenize(text: str) -> set[str]:
    text = text.lower()
    all_words = re.findall("[a-z0-9]+", text)  # extract the words
    return set(all_words)


class NaiveBayes(SpamModel):
    def __init__(self, k: float = 0.5) -> None:
        self.k = k
        self.classify_fn: SpamClassifier = None

    def train(self, dataset: Dataset) -> SpamClassifier:
        k = self.k
        dataset_tokens: set[str] = set()
        token_spam_counts: dict[str, int] = defaultdict(int)
        token_ham_counts: dict[str, int] = defaultdict(int)
        spam_messages = ham_messages = 0

        for example in dataset:
            if example.spam:
                spam_messages += 1
            else:
                ham_messages += 1

            # Increment word counts
            for token in tokenize(example.email):
                dataset_tokens.add(token)
                if example.spam:
                    token_spam_counts[token] += 1
                else:
                    token_ham_counts[token] += 1

        def classify(email: str) -> Prediction:
            email_tokens = tokenize(email)
            log_prob_if_spam = log_prob_if_ham = 0.0

            # Iterate through each word in our vocabulary
            for token in dataset_tokens:
                spam = token_spam_counts[token]
                ham = token_ham_counts[token]

                prob_if_spam = (spam + k) / (spam_messages + 2 * k)
                prob_if_ham = (ham + k) / (ham_messages + 2 * k)
                # If *token* appears in the message,
                # add the log probability of seeing it
                if token in email_tokens:
                    log_prob_if_spam += math.log(prob_if_spam)
                    log_prob_if_ham += math.log(prob_if_ham)
                # Otherwise add the log probability of _not_ seeing it,
                # which is log(1 - probability of seeing it)
                else:
                    log_prob_if_spam += math.log(1.0 - prob_if_spam)
                    log_prob_if_ham += math.log(1.0 - prob_if_ham)
            prob_if_spam = math.exp(log_prob_if_spam)
            prob_if_ham = math.exp(log_prob_if_ham)
            return prob_if_spam / (prob_if_spam + prob_if_ham)

        return classify

    def load(self, sha256_digest: str, model_registry_root: pathlib.Path) -> None:
        self.classify_fn = model_trainer.load_serialized_classifier(
            classifier_sha256_hash=sha256_digest,
            classifier_destination_root=model_registry_root,
        )

    def save(self, fn: SpamClassifier, model_registry_root: pathlib.Path) -> str:
        return model_trainer.store_classifier(
            classifier_func=fn,
            classifier_destination_root=model_registry_root,
            current_git_commit_hash="ffofofo",
        )
