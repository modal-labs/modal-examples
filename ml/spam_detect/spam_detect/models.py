import math
import pathlib
import re
from collections import defaultdict

from .datasets.enron import structure
from . import config
from . import model_trainer

from typing import (
    cast,
    Callable,
    Iterable,
    Protocol,
)

Prediction = float
Dataset = Iterable[structure.Example]
SpamClassifier = Callable[[str], Prediction]


def tokenize(text: str) -> set[str]:
    text = text.lower()
    all_words = re.findall("[a-z0-9]+", text)  # extract the words
    return set(all_words)


class SpamModel(Protocol):
    """The training and storage interface that all spam-classification models must implement."""

    def train(self, dataset: Dataset) -> SpamClassifier:
        ...

    def load(self, sha256_digest: str, model_registry_root: pathlib.Path) -> SpamClassifier:
        ...

    def save(self, fn: SpamClassifier, model_path: pathlib.Path) -> str:
        ...


def construct_huggingface_dataset(dataset: Dataset):
    import datasets
    import pyarrow as pa

    emails = pa.array((ex.email for ex in dataset), type=pa.string())
    labels = pa.array((ex.spam for ex in dataset), type=pa.bool_())
    pa_table = pa.table([emails, labels], names=["text", "labels"])
    return datasets.Dataset(pa_table).train_test_split(test_size=0.1)


def train_llm_classifier(dataset: Dataset, dry_run: bool = True):
    import numpy as np
    import evaluate
    import pyarrow
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification
    from transformers import AutoTokenizer
    from transformers import TrainingArguments, Trainer

    logger = config.get_logger()

    # dataset_ = load_dataset("yelp_review_full")
    # dataset_["train"][100]
    huggingface_dataset = construct_huggingface_dataset(dataset)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # import IPython
    # IPython.embed()

    # return

    tokenized_datasets = huggingface_dataset.map(tokenize_function, batched=True)

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

    if not dry_run:
        trainer.train()
        logger.opt(colors=True).info(f"<light-green>✔️ training done!</light-green>")
    else:
        logger.info(f"{dry_run=}, so skipping training step.")

    return trainer


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
        print(self.model.config.id2label[predicted_class_id])
        # TODO(Jonathon): map predicted class to boolean.
        return 0.5


class LLM(SpamModel):
    """
    A large-language model (LLM) fine-tuned for the SPAM/HAM text classification problem.

    Uses huggingface/transformers library.
    """

    model_name = "bert-base-cased"

    def train(self, dataset: Dataset) -> SpamClassifier:
        from transformers import AutoTokenizer

        trainer = train_llm_classifier(dataset=dataset)
        model = trainer.model
        tokenizer = AutoTokenizer.from_pretrained(LLM.model_name)
        return LLMSpamClassifier(
            tokenizer=tokenizer,
            model=model,
        )

    def load(self, sha256_digest: str, model_registry_root: pathlib.Path) -> SpamClassifier:
        from transformers import AutoTokenizer
        from transformers import AutoModelForSequenceClassification

        # TODO: refactor to use model_trainer module for loading.
        model_path = model_registry_root / sha256_digest
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(LLM.model_name)
        return LLMSpamClassifier(
            tokenizer=tokenizer,
            model=model,
        )

    def save(self, fn: SpamClassifier, model_registry_root: pathlib.Path) -> str:
        from transformers import Trainer

        llm_fn = cast(LLMSpamClassifier, fn)
        trainer = Trainer(model=llm_fn.model)
        return model_trainer.store_huggingface_model(
            trainer=trainer,
            model_name=LLM.model_name,
            model_destination_root=model_registry_root,
            git_commit_hash="foobar",
        )


class BadWords(SpamModel):
    """
    An extremely rudimentary heuritistic model. If a trained model
    can't beat this something is very wrong.
    """

    def train(self, dataset: Dataset) -> SpamClassifier:
        _ = dataset

        def bad_words_spam_classifier(email: str) -> Prediction:
            tokens = " ".split(email)
            tokens_set = set(tokens)
            # TODO: investigate is using a set here makes serialization non-deterministic.
            bad_words = [
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
            return 1.0 if bad_words_count > max_bad_words else 0.0

        return bad_words_spam_classifier

    def load(self, sha256_digest: str, model_registry_root: pathlib.Path) -> SpamClassifier:
        return model_trainer.load_pickle_serialized_model(
            classifier_sha256_hash=sha256_digest,
            classifier_destination_root=model_registry_root,
        )

    def save(self, fn: SpamClassifier, model_registry_root: pathlib.Path) -> str:
        return model_trainer.store_picklable_model(
            classifier_func=fn,
            classifier_destination_root=model_registry_root,
            current_git_commit_hash="ffofofo",
        )


class NaiveBayes(SpamModel):
    """
    The classic Naive-Bayes classifier. Implementation drawn from the
    *Data Science From Scratch* book: github.com/joelgrus/data-science-from-scratch.
    """

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

    def load(self, sha256_digest: str, model_registry_root: pathlib.Path) -> SpamClassifier:
        return model_trainer.load_pickle_serialized_model(
            classifier_sha256_hash=sha256_digest,
            classifier_destination_root=model_registry_root,
        )

    def save(self, fn: SpamClassifier, model_registry_root: pathlib.Path) -> str:
        return model_trainer.store_picklable_model(
            classifier_func=fn,
            classifier_destination_root=model_registry_root,
            current_git_commit_hash="ffofofo",
        )
