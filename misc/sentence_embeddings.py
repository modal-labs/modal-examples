# ---
# lambda-test: false
# ---
"""
Computes sentence embeddings using Modal.
This uses huggingface's `transformer` library to calculate the vector
representations for a collection of sentences.

Example modified from: https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1

Install dependencies before running example:

$ pip3 install torch==1.10.2 tqdm numpy requests tensorboard
"""
import tarfile
from pathlib import Path
from typing import List

import modal

# dependencies
dependencies = ["torch==1.10.2", "transformers==4.16.2", "tensorboard"]
stub = modal.Stub(image=modal.Image.debian_slim().pip_install(dependencies))

if stub.is_inside():
    import numpy as np
    import requests
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from transformers import AutoModel, AutoTokenizer

    TOKENIZER = AutoTokenizer.from_pretrained(
        "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
    )
    MODEL = AutoModel.from_pretrained(
        "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
    )


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


@stub.function
def vectorize(x: str):
    """Vectorizes a string (sentence) into a PyTorch Tensor."""
    # encode input and calculate vector
    encoded_input = TOKENIZER(x, padding=True, truncation=True, return_tensors="pt")
    model_output = MODEL(**encoded_input)
    y = mean_pooling(model_output, encoded_input["attention_mask"])

    sentence = f"{x[:100]} ..."
    print(f" → calculated vector for: {sentence}")
    return (sentence, y.detach().numpy())


def write_tensorboard_logs(embedding, metadata: List[str]):
    """Write tensorboard logs."""
    print(" → writing tensorboard logs")
    writer = SummaryWriter(log_dir="./tensorboard")
    writer.add_embedding(np.concatenate(embedding), metadata)
    writer.close()


class MovieReviewsDataset:
    """Standford's Large Movie Review Dataset."""

    url: str = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    def __init__(self, output_path: Path = Path("./movie_reviews_dataset.tar.gz")):
        self.output_path = output_path
        self.dataset: List[str] = []

    def download(self) -> None:
        """Downloads dataset if not available locally."""
        if self.output_path.exists():
            print(" → using cached dataset")
            return

        print(f" → downloading dataset from {self.url}")
        with requests.get(self.url, stream=True) as r:
            r.raise_for_status()
            with open(self.output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    # if chunk:
                    f.write(chunk)

        print(" → dataset downloaded")

    def extract(self) -> None:
        """Extracts dataset package into local path."""
        if Path("./aclImdb").exists():
            print(" → using cached extracted dataset")
            return

        print(f" → extracting file {self.output_path}")
        with tarfile.open(self.output_path) as tf:
            tf.extractall()

    def index(self) -> None:
        """Creates in-memory index for dataset"""
        # read only train files
        dataset_path = Path("./aclImdb/train")

        # combine both negative and positive files
        negative = [f for f in (dataset_path / "neg").glob("*.txt")]
        positive = [f for f in (dataset_path / "pos").glob("*.txt")]
        return negative + positive

    def load(self, sample_size: int = 100) -> None:
        """Loads dataset into memory"""
        print(f" → loading dataset with sample size = {sample_size}")
        for file in np.random.choice(self.index(), size=sample_size):
            self.dataset.append(file.open().read())  # limit to 500 characters


if __name__ == "__main__":

    # download & prepare dataset
    dataset = MovieReviewsDataset()
    dataset.download()
    dataset.extract()

    # load dataset into memory; use a small samplesize (N=25k)
    dataset.load(sample_size=100)

    # vectorize the entire dataset
    embedding = []
    metadata = []
    with stub.run():
        for sentence, vector in vectorize.map(dataset.dataset):
            embedding.append(vector)
            metadata.append(sentence)

        write_tensorboard_logs(embedding, metadata)

    # open tensorboard
    print(" → done!")
    print(" → to see results in TensorBoard, run: tensorboard --logdir tensorboard/")
    print(" → (open http://localhost:6006#projector and wait for it to load)")
