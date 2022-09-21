# ---
# integration-test: false
# ---
# # Batch inference using a model from Huggingface
#
# <center>
#   <img src="./batch_inference_huggingface.png"/>
# </center>
#
# This example shows how to use a sentiment analysis model from Huggingface to classify
# 25,000 movie reviews in a couple of minutes.
#
# Some Modal features it uses:
# * Container lifecycle hook: this lets us load the model only once in each container
# * CPU requests: the prediction function is very CPU-hungry, so we reserve 8 cores
# * Mapping: we map over 25,000 sentences and Modal manages the pool of containers for us
#
# ## Basic setup
#
# Let's get started writing code.
# For the Modal container image, we need a few Python packages,
# including `transformers`, which is the main Huggingface package.

import io

import modal

stub = modal.Stub(
    image=modal.Image.debian_slim().pip_install(
        ["datasets", "matplotlib", "sklearn", "torch", "transformers"]
    )
)

# ## Defining the prediction function
#
# Instead of a using `@stub.function` in the global scope,
# we put the method on a class, and define an `__enter__` method on that class.
# Modal reuses containers for successive calls to the same function, so
# we want to take advantage of this and avoid setting up the same model
# for every function call.
#
# Since the transformer model is very CPU-hungry, we allocate 8 CPUs
# to the model.
# Every container that runs will have 8 CPUs set aside for it.


class SentimentAnalysis:
    def __enter__(self):
        from transformers import pipeline

        self.sentiment_pipeline = pipeline(
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    @stub.function(cpu=8)
    def predict(self, phrase: str):
        pred = self.sentiment_pipeline(phrase, truncation=True, max_length=512, top_k=2)
        # pred will look like: [{'label': 'NEGATIVE', 'score': 0.99}, {'label': 'POSITIVE', 'score': 0.01}]
        probs = {p["label"]: p["score"] for p in pred}
        return probs["POSITIVE"]


# ## Getting data
#
# We need some data to run the batch inference on.
# We use this [dataset of IMDB reviews](https://ai.stanford.edu/~amaas/data/sentiment/) for this purpose.
# Huggingface actually offers this data [as a preprocessed dataaset](https://huggingface.co/datasets/imdb),
# which we can download using the `datasets` package:


@stub.function
def get_data():
    from datasets import load_dataset

    imdb = load_dataset("imdb")
    data = [(row["text"], row["label"]) for row in imdb["test"]]
    return data


# ## Plotting the ROC curve
#
# In order to evaluate the classifier, let's plot an
# [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic).
# This is a common way to evaluate classifiers on binary data.


@stub.function
def roc_plot(labels, predictions):
    from matplotlib import pyplot
    from sklearn.metrics import RocCurveDisplay

    pyplot.style.use("ggplot")
    RocCurveDisplay.from_predictions(labels, predictions)
    buf = io.BytesIO()
    pyplot.savefig(buf, format="png")
    return buf.getvalue()


# A bit of a spoiler warning, but if you run this script, the ROC curve will look like this:
#
# ![roc](./batch_inference_roc.png)
#
# The AUC of this classifier is 0.96, which means it's very good!

# ## Putting it together
#
# The main flow of the code downloads the data, then runs the batch inference,
# then plots the results.
# Each prediction takes roughly 0.1-1s, so if we ran everything sequentially it would take 2,500-25,000 seconds.
# That's a lot! Luckily because of Modal's `.map` method, we can process everything in a couple of minutes at most.
# Modal will automatically spin up more and more workers until all inputs are processed.

if __name__ == "__main__":
    with stub.run():
        print("Downloading data...")
        data = get_data()
        print("Got", len(data), "reviews")
        reviews = [review for review, label in data]
        labels = [label for review, label in data]

        # Let's check that the model works by classifying the first 5 entries
        predictor = SentimentAnalysis()
        for review, label in data[:5]:
            prediction = predictor.predict(review)
            print(
                f"Sample prediction with positivity score {prediction}:\n{review}\n\n"
            )

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

# ## Running this
#
# When you run this, it will download the dataset and load the model, then output some
# sample predictions:
#
# After that, it kicks off the actual batch inference.
# It should look something like the screenshot below (we are very proud of the progress bar):
#
# ![progress](./batch_inference_progress.png)
#
# The whole thing should take a few minutes to run.
#
# ## Further optimization notes
#
# Every container downloads the model when it starts, which is a bit inefficient.
# In order to improve this, what you could do is to set up a shared volume that gets
# mounted to each container.
# See [shared volumes](docs/guide/shared-volumes).
#
# In order for Huggingface to use the shared volume, you need to set the value of
# the `TRANSFORMERS_CACHE` environment variable to the path of the shared volume.
# See [secrets](/docs/guide/secrets).
