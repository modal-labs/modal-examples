# ---
# integration-test: false
# ---
# # Batch inference using a model from Huggingface
#
# ![huggingface](./batch_inference_huggingface.png)
#
# This example shows how to use a sentiment analysis model from Huggingface to classify 25,000 movie ratings.
#
# Some Modal features it uses:
# * Container lifecycle hook: this lets us load the model once in each container
# * CPU requests: the prediction function is very CPU-hungry, so we reserve 8 cores
# * Mapping: we map over 25,000 sentences in about a minute or less
#
# ## Basic setup
#
# Global imports:

import io

import modal

# Next, let's set up the Modal environment.
# All the Python packages, as well as the shared volume and the environment variables

stub = modal.Stub(
    image=modal.Image.debian_slim().pip_install(
        ["datasets", "matplotlib", "sklearn", "torch", "transformers"]
    )
)

# ## Defining the prediction function
#
# The prediction function uses a few features with Modal:
#
# Instead of a global function, we put the method on a class,
# and define an `__enter__` method on that class.
# This method will be executed only once for each container.
# The point of this is to load the model into memory only once, since
# this is a slow operaton (a few seconds).


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
# We use this [online dataset of movie reviews](https://ai.stanford.edu/~amaas/data/sentiment/) for this purpose.
# As it turns out, Huggingface also [hosts this data](https://huggingface.co/datasets/imdb)


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
#
# Spoiling the end, the output of this script will look like this:
#
# ![roc](batch_inference_roc.png)


@stub.function
def roc_plot(labels, predictions):
    from matplotlib import pyplot
    from sklearn.metrics import RocCurveDisplay

    pyplot.style.use("ggplot")
    RocCurveDisplay.from_predictions(labels, predictions)
    buf = io.BytesIO()
    pyplot.savefig(buf, format="png")
    return buf.getvalue()


# ## Putting it together
#
# The main flow of the code downloads the data, then runs the batch inference,
# then plots the results.
# Each prediction takes roughly 0.1-1s, so if we ran everything sequentially it would take 2,500-25,000 seconds.
# That's a lot! Luckily because of Modal's `.map` method, we can process everything in a couple of minutes at most.

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
# When you run this, you should see something like this:
#
# <center>
# <video controls>
# <source src="./batch_inference_screen.mp4" type="video/mp4">
# <track kind="captions" />
# </video>
# </center>
#
# ## Further optimization notes
#
# Every container downloads the model when it starts, which is a bit inefficient.
# In order to improve this, what you could do is to set up a shared volume that gets
# mounted to each container.
# You have to use that in conjunction with the `TRANSFORMERS_CACHE` environment variable
# to tell Huggingface where to store the model.
