import modal
import tarfile
import urllib.request as urllib2

HUGGINGFACE_DIR = "/huggingface"
stub = modal.Stub(
    image=modal.Image.debian_slim().pip_install(["torch", "transformers"])
)
stub.sv = modal.SharedVolume()

env = modal.Secret({"TRANSFORMERS_CACHE": HUGGINGFACE_DIR})


class SentimentAnalysis:
    def __enter__(self):
        from transformers import pipeline

        print("loading pipeline...")
        self.sentiment_pipeline = pipeline(
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        print("ready!")

    @stub.function(
        shared_volumes={HUGGINGFACE_DIR: stub.sv}, secrets=[env], concurrency_limit=25
    )
    def predict(self, phrase: str):
        (pred,) = self.sentiment_pipeline(phrase, truncation=True, max_length=512)

        # Return the positive score
        if pred["label"] == "POSITIVE":
            return pred["score"]
        else:
            return 1 - pred["score"]


@stub.function
def run():
    # First, read reviews dataset
    print("reading dataset")
    rt = urllib2.urlopen(
        "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    )
    tf = tarfile.open(fileobj=rt, mode="r:gz")
    labels = []
    reviews = []
    for member in tf:
        if member.name.startswith("aclImdb/test/pos/"):
            label = 0
        elif member.name.startswith("aclImdb/test/neg/"):
            label = 1
        else:
            continue
        with tf.extractfile(member) as tfobj:
            review = tfobj.read().decode()
            labels.append(label)
            reviews.append(review)
    print("got", len(reviews), "reviews")

    # In order to force the model to be downloaded only once, run a dummy predictor
    # Otherwise, the model will be downloaded by every worker in the map(...)
    predictor = SentimentAnalysis()
    predictor.predict("test")

    # Now, let's run batch inference over it
    for ret in predictor.predict.map(reviews):
        print(ret)


if __name__ == "__main__":
    with stub.run():
        run()
