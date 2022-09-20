# ---
# integration-test: false
# ---

import io
import random
import tarfile
import urllib.request as urllib2

import modal

HUGGINGFACE_DIR = "/huggingface"
stub = modal.Stub(
    image=modal.Image.debian_slim().pip_install(
        ["matplotlib", "sklearn", "torch", "transformers"]
    )
)
stub.sv = modal.SharedVolume()

env = modal.Secret({"TRANSFORMERS_CACHE": HUGGINGFACE_DIR})


class SentimentAnalysis:
    def __enter__(self):
        from transformers import pipeline

        self.sentiment_pipeline = pipeline(
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    @stub.function(
        shared_volumes={HUGGINGFACE_DIR: stub.sv},
        secrets=[env],
        cpu=3,  # TODO: bump CPU
    )
    def predict(self, phrase: str):
        pred = self.sentiment_pipeline(phrase, truncation=True, max_length=512, top_k=2)
        # pred will look like: [{'label': 'NEGATIVE', 'score': 0.99}, {'label': 'POSITIVE', 'score': 0.01}]
        probs = {p["label"]: p["score"] for p in pred}
        return probs["POSITIVE"]


@stub.function
def get_data():
    rt = urllib2.urlopen(
        "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    )
    tf = tarfile.open(fileobj=rt, mode="r:gz")
    data = []
    for member in tf:
        if member.name.startswith("aclImdb/test/pos/"):
            label = 1
        elif member.name.startswith("aclImdb/test/neg/"):
            label = 0
        else:
            continue
        with tf.extractfile(member) as tfobj:
            review = tfobj.read().decode()
            data.append((label, review))

    random.shuffle(data)
    return data


@stub.function
def roc_plot(labels, predictions):
    from matplotlib import pyplot
    from sklearn.metrics import RocCurveDisplay

    pyplot.style.use("ggplot")
    RocCurveDisplay.from_predictions(labels, predictions)
    buf = io.BytesIO()
    pyplot.savefig(buf, format="png")
    return buf.getvalue()


if __name__ == "__main__":
    with stub.run():
        print("Downloading data...")
        data = get_data()
        # data = data[:100]
        print("Got", len(data), "reviews")
        reviews = [review for label, review in data]
        labels = [label for label, review in data]

        # In order to force the model to be downloaded only once, run a dummy predictor
        # Otherwise, the model will be downloaded by multiple workers starting simultaneously
        print("Downloading model...")
        predictor = SentimentAnalysis()
        predictor.predict("test")

        # Now, let's run batch inference over it
        print("Running batch prediction...")
        predictions = list(predictor.predict.map(reviews))

        # Generate a ROC plot
        print("Creating ROC plot...")
        png_data = roc_plot(labels, predictions)
        fn = "/tmp/roc.png"
        with open(fn, "wb") as f:
            f.write(png_data)
        print(f"Wrote ROC curve to {fn}")
