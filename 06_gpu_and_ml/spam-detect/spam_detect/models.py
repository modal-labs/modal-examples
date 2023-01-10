"""

The core model interface is `SpamModel`, which must be implemented by all
trainable and serveable spam-detection models in the module.

Current model implementations are:

* BadWords (a baseline heuristic classifier)
* LLM (a fine-tuned BERT language classifier)
* NaiveBayes
"""
import json
import math
import pathlib
import re
from collections import defaultdict

from . import config
from . import dataset
from . import model_storage
from .model_registry import ModelMetadata, TrainMetrics

from typing import (
    cast,
    Callable,
    Iterable,
    NamedTuple,
    Optional,
    Protocol,
)


class Prediction(NamedTuple):
    spam: bool
    score: float


Dataset = Iterable[dataset.Example]
SpamClassifier = Callable[[str], Prediction]


def load_model(model_id: str):
    registry_filepath = config.MODEL_STORE_DIR / config.MODEL_REGISTRY_FILENAME
    with open(registry_filepath, "r") as f:
        registry_data = json.load(f)
    if model_id not in registry_data:
        raise ValueError(f"{model_id} not contained in registry.")

    metadata = ModelMetadata.from_dict(registry_data[model_id])
    if metadata.impl_name == "bert-base-cased":
        m = LLM()
    elif "NaiveBayes" in metadata.impl_name:
        m = NaiveBayes()
    else:
        raise ValueError(f"Loading '{metadata.impl_name}' not yet supported.")

    classifier = m.load(sha256_digest=config.SERVING_MODEL_ID, model_registry_root=config.MODEL_STORE_DIR)
    return classifier, metadata


def tokenize(text: str) -> set[str]:
    text = text.lower()
    all_words = re.findall("[a-z0-9]+", text)  # extract the words
    return set(all_words)


class SpamModel(Protocol):
    """The training and storage interface that all spam-classification models must implement."""

    def train(self, dataset: Dataset) -> tuple[SpamClassifier, TrainMetrics]:
        ...

    def load(self, sha256_digest: str, model_registry_root: pathlib.Path) -> SpamClassifier:
        ...

    def save(self, fn: SpamClassifier, metrics: TrainMetrics, model_registry_root: pathlib.Path) -> str:
        ...


def construct_huggingface_dataset(dataset: Dataset, label2id: dict[str, int]):
    import datasets
    import pyarrow as pa

    emails = pa.array((ex.email for ex in dataset), type=pa.string())
    labels = pa.array((label2id["SPAM"] if ex.spam else label2id["HAM"] for ex in dataset), type=pa.uint8())
    pa_table = pa.table([emails, labels], names=["text", "labels"])
    return datasets.Dataset(pa_table).train_test_split(test_size=0.1)


class LLMSpamClassifier:
    """SpamClassifier that wraps a fine-tuned Huggingface BERT transformer model."""

    def __init__(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, email: str) -> Prediction:
        """Ensures this class-based classifier can be used just like a function-based classifer."""
        import torch

        inputs = self.tokenizer(email, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_class_id = logits.argmax().item()
        spam_id = self.model.config.label2id["SPAM"]
        spam_score = logits[0][spam_id]
        predicted_label: str = self.model.config.id2label[predicted_class_id]
        return Prediction(
            spam=bool(predicted_label == "SPAM"),
            score=spam_score,
        )


def train_llm_classifier(dataset: Dataset, dry_run: bool = False) -> tuple[LLMSpamClassifier, TrainMetrics]:
    import numpy as np
    import evaluate
    import pyarrow
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification
    from transformers import AutoTokenizer
    from transformers import TrainingArguments, Trainer

    logger = config.get_logger()

    id2label = {0: "HAM", 1: "SPAM"}
    label2id = {"HAM": 0, "SPAM": 1}
    huggingface_dataset = construct_huggingface_dataset(dataset, label2id)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = huggingface_dataset.map(tokenize_function, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

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

    if not dry_run:
        trainer.train()
        logger.opt(colors=True).info(f"<light-green>✔️ training done!</light-green>")
    else:
        logger.info(f"{dry_run=}, so skipping training step.")

    metrics = TrainMetrics(
        dataset_id="enron",
        eval_set_size=-1,
        accuracy=0.0,
    )
    return trainer, metrics


class LLM(SpamModel):
    """
    A large-language model (LLM) fine-tuned for the SPAM/HAM text classification problem.

    Uses huggingface/transformers library.
    """

    model_name = "bert-base-cased"

    def train(self, dataset: Dataset) -> tuple[SpamClassifier, TrainMetrics]:
        from transformers import AutoTokenizer

        trainer, metrics = train_llm_classifier(dataset=dataset)
        model = trainer.model
        tokenizer = AutoTokenizer.from_pretrained(LLM.model_name)
        return (
            LLMSpamClassifier(
                tokenizer=tokenizer,
                model=model,
            ),
            metrics,
        )

    def load(self, sha256_digest: str, model_registry_root: pathlib.Path) -> SpamClassifier:
        from transformers import AutoTokenizer
        from transformers import AutoModelForSequenceClassification

        # TODO: refactor to use model_storage module for loading.
        model_path = model_registry_root / sha256_digest
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(LLM.model_name)
        return LLMSpamClassifier(
            tokenizer=tokenizer,
            model=model,
        )

    def save(self, fn: SpamClassifier, metrics: TrainMetrics, model_registry_root: pathlib.Path) -> str:
        from transformers import Trainer

        llm_fn = cast(LLMSpamClassifier, fn)
        trainer = Trainer(model=llm_fn.model)
        return model_storage.store_huggingface_model(
            trainer=trainer,
            train_metrics=metrics,
            model_name=LLM.model_name,
            model_destination_root=model_registry_root,
            git_commit_hash="foobar",
        )


class BadWords(SpamModel):
    """
    An extremely rudimentary heuritistic model. If a trained model
    can't beat this something is very wrong.
    """

    def train(self, dataset: Dataset) -> tuple[SpamClassifier, TrainMetrics]:
        def bad_words_spam_classifier(email: str) -> Prediction:
            tokens = email.split(" ")
            tokens_set = set(tokens)
            # NB: using a set here makes pickle serialization non-deterministic.
            bad_words = [
                "click",  # http://www.paulgraham.com/spam.html
                "sex",
                "xxx",
                "nigerian",
                "teens",
            ]
            max_bad_words = 2
            bad_words_count = 0
            for word in bad_words:
                if word in tokens_set:
                    bad_words_count += 1
            return (
                Prediction(score=1.0, spam=True)
                if bad_words_count > max_bad_words
                else Prediction(score=0.0, spam=False)
            )

        accuracy, precision = self._calc_metrics(classifier=bad_words_spam_classifier, dataset=dataset)
        metrics = TrainMetrics(
            dataset_id="enron",
            eval_set_size=0,
            accuracy=accuracy,
            precision=precision,
        )
        return bad_words_spam_classifier, metrics

    def load(self, sha256_digest: str, model_registry_root: pathlib.Path) -> SpamClassifier:
        return model_storage.load_pickle_serialized_model(
            sha256_hash=sha256_digest,
            destination_root=model_registry_root,
        )

    def save(self, fn: SpamClassifier, metrics: TrainMetrics, model_registry_root: pathlib.Path) -> str:
        return model_storage.store_pickleable_model(
            classifier_func=fn,
            metrics=metrics,
            model_destination_root=model_registry_root,
            current_git_commit_hash="ffofofo",
        )

    def _calc_metrics(self, classifier: SpamClassifier, dataset: Dataset) -> tuple[int, int]:
        if len(dataset) == 0:
            raise ValueError("Evaluation dataset cannot be empty.")
        tp, tn, fp, fn = 0, 0, 0, 0
        for example in dataset:
            pred = classifier(example.email)
            if pred.spam and example.spam:
                tp += 1
            elif pred.spam and not example.spam:
                fp += 1
            elif not pred.spam and not example.spam:
                tn += 1
            else:
                fn += 1
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        print(f"Summary: {tp=} {fp=} {tn=} {fn=}")
        return accuracy, precision


class NaiveBayes(SpamModel):
    """
    The classic Naive-Bayes classifier. Implementation drawn from the
    *Data Science From Scratch* book: github.com/joelgrus/data-science-from-scratch.
    """

    def __init__(self, k: float = 0.5, decision_boundary: Optional[float] = None) -> None:
        self.k = k
        self.decision_boundary = decision_boundary
        self.classify_fn: SpamClassifier = None

    def train(self, dataset: Dataset) -> tuple[SpamClassifier, TrainMetrics]:
        k = self.k
        dataset_tokens: set[str] = set()
        token_spam_counts: dict[str, int] = defaultdict(int)
        token_ham_counts: dict[str, int] = defaultdict(int)
        spam_messages = ham_messages = 0
        test_size = 0.05
        train_set = dataset[: int(len(dataset) * test_size)]
        test_set = dataset[-int(len(dataset) * test_size) :]

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

        print("finished building word count dicts")

        def predict_prob(email: str) -> float:
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
            score = prob_if_spam / (prob_if_spam + prob_if_ham) if prob_if_spam else 0.0
            return score

        def make_classifier(prob_fn, decision_boundary: float) -> SpamClassifier:
            def inner(email: str):
                score = prob_fn(email)
                return Prediction(
                    spam=score > decision_boundary,
                    score=score,
                )

            return inner

        if self.decision_boundary:
            decision_boundary, precision, recall = self.decision_boundary, None, None
        else:
            print("setting decision boundary for binary classifier")
            decision_boundary, precision, recall = self._set_decision_boundary(
                prob_fn=predict_prob,
                test_dataset=test_set,
            )

        metrics = TrainMetrics(
            dataset_id="enron",
            eval_set_size=len(test_set),
            accuracy=None,
            precision=precision,
            recall=recall,
        )
        print("making classifier")
        return make_classifier(predict_prob, decision_boundary), metrics

    def load(self, sha256_digest: str, model_registry_root: pathlib.Path) -> SpamClassifier:
        return model_storage.load_pickle_serialized_model(
            sha256_hash=sha256_digest,
            destination_root=model_registry_root,
        )

    def save(self, fn: SpamClassifier, metrics: TrainMetrics, model_registry_root: pathlib.Path) -> str:
        return model_storage.store_pickleable_model(
            classifier_func=fn,
            metrics=metrics,
            model_destination_root=model_registry_root,
            current_git_commit_hash="ffofofo",
        )

    def _set_decision_boundary(self, prob_fn, test_dataset) -> float:
        import numpy as np
        from sklearn.metrics import precision_recall_curve

        print(f"Using {len(test_dataset)} test dataset examples to set decision boundary")

        minimum_acceptable_precision = 0.98  # ie. 2 in a 100 legit emails get marked as spam.
        y_true = np.array([1 if ex.spam else 0 for ex in test_dataset])
        # scores are rounded because curve calculation time scales quickly in dim U, where U is number of unique scores.
        # NB: The precision-recall curve calculation is extremely slow on N ~10k+
        y_scores = np.array([round(prob_fn(ex.email), ndigits=2) for ex in test_dataset])
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        for p, r, thres in zip(precisions, recalls, thresholds):
            print("Using threshold={} as decision boundary, we reach precision={} and recall={}".format(thres, p, r))
            if p >= minimum_acceptable_precision:
                print(f"Reached {minimum_acceptable_precision=} at threshold {thres}. Setting that as boundary.")
                break
        return thres, p, r
