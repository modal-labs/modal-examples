# ---
# runtimes: ["runc", "gvisor"]
# ---
# # Batch inference using a model from Huggingface
#
# <center>
#   <img src="./batch_inference_huggingface.png" alt="Huggingface company logo" />
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

app = modal.App(
    "example-batch-inference-using-huggingface",
    image=modal.Image.debian_slim().pip_install(
        "datasets",
        "matplotlib",
        "scikit-learn",
        "torch",
        "transformers",
    ),
)

# ## Defining the prediction function
#
# Instead of a using `@app.function()` in the global scope,
# we put the method on a class, and define a setup method that we
# decorate with `@modal.enter()`.
#
# Modal reuses containers for successive calls to the same function, so
# we want to take advantage of this and avoid setting up the same model
# for every function call.
#
# Since the transformer model is very CPU-hungry, we allocate 8 CPUs
# to the model. Every container that runs will have 8 CPUs set aside for it.


@app.cls(cpu=8, retries=3)
class SentimentAnalysis:
    @modal.enter()
    def setup_pipeline(self):
        from transformers import pipeline

        self.sentiment_pipeline = pipeline(
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    @modal.method()
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


@app.function()
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


@app.function()
def roc_plot(labels, predictions):
    from matplotlib import pyplot
    from sklearn.metrics import RocCurveDisplay

    pyplot.style.use("ggplot")
    RocCurveDisplay.from_predictions(labels, predictions)
    with io.BytesIO() as buf:
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


@app.local_entrypoint()
def main():
    print("Downloading data...")
    data = get_data.remote()
    print("Got", len(data), "reviews")
    reviews = [review for review, label in data]
    labels = [label for review, label in data]

    # Let's check that the model works by classifying the first 5 entries
    predictor = SentimentAnalysis()
    for review, label in data[:5]:
        prediction = predictor.predict.remote(review)
        print(f"Sample prediction with positivity score {prediction}:\n{review}\n\n")

    # Now, let's run batch inference over it
    print("Running batch prediction...")
    predictions = list(predictor.predict.map(reviews))

    # Generate a ROC plot
    print("Creating ROC plot...")
    png_data = roc_plot.remote(labels, predictions)
    fn = "/tmp/roc.png"
    with open(fn, "wb") as f:
        f.write(png_data)
    print(f"Wrote ROC curve to {fn}")


# ## Running this
#
# When you run this, it will download the dataset and load the model, then output some
# sample predictions:
#
# ```
# Sample prediction with positivity score 0.0003837468393612653:
# I love sci-fi and am willing to put up with a lot. Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood. I tried to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Silly prosthetics, cheap cardboard sets, stilted dialogues, CG that doesn't match the background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi' setting. (I'm sure there are those of you out there who think Babylon 5 is good sci-fi TV. It's not. It's clichéd and uninspiring.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may treat important issues, yet not as a serious philosophy. It's really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are wooden and predictable, often painful to watch. The makers of Earth KNOW it's rubbish as they have to always say "Gene Roddenberry's Earth..." otherwise people would not continue watching. Roddenberry's ashes must be turning in their orbit as this dull, cheap, poorly edited (watching it without advert breaks really brings this home) trudging Trabant of a show lumbers into space. Spoiler. So, kill off a main character. And then bring him back as another actor. Jeeez! Dallas all over again.
#
# Sample prediction with positivity score 0.38294079899787903:
# Worth the entertainment value of a rental, especially if you like action movies. This one features the usual car chases, fights with the great Van Damme kick style, shooting battles with the 40 shell load shotgun, and even terrorist style bombs. All of this is entertaining and competently handled but there is nothing that really blows you away if you've seen your share before.<br /><br />The plot is made interesting by the inclusion of a rabbit, which is clever but hardly profound. Many of the characters are heavily stereotyped -- the angry veterans, the terrified illegal aliens, the crooked cops, the indifferent feds, the bitchy tough lady station head, the crooked politician, the fat federale who looks like he was typecast as the Mexican in a Hollywood movie from the 1940s. All passably acted but again nothing special.<br /><br />I thought the main villains were pretty well done and fairly well acted. By the end of the movie you certainly knew who the good guys were and weren't. There was an emotional lift as the really bad ones got their just deserts. Very simplistic, but then you weren't expecting Hamlet, right? The only thing I found really annoying was the constant cuts to VDs daughter during the last fight scene.<br /><br />Not bad. Not good. Passable 4.
#
# Sample prediction with positivity score 0.0002899310493376106:
# its a totally average film with a few semi-alright action sequences that make the plot seem a little better and remind the viewer of the classic van dam films. parts of the plot don't make sense and seem to be added in to use up time. the end plot is that of a very basic type that doesn't leave the viewer guessing and any twists are obvious from the beginning. the end scene with the flask backs don't make sense as they are added in and seem to have little relevance to the history of van dam's character. not really worth watching again, bit disappointed in the end production, even though it is apparent it was shot on a low budget certain shots and sections in the film are of poor directed quality
#
# Sample prediction with positivity score 0.004243704490363598:
# STAR RATING: ***** Saturday Night **** Friday Night *** Friday Morning ** Sunday Night * Monday Morning <br /><br />Former New Orleans homicide cop Jack Robideaux (Jean Claude Van Damme) is re-assigned to Columbus, a small but violent town in Mexico to help the police there with their efforts to stop a major heroin smuggling operation into their town. The culprits turn out to be ex-military, lead by former commander Benjamin Meyers (Stephen Lord, otherwise known as Jase from East Enders) who is using a special method he learned in Afghanistan to fight off his opponents. But Jack has a more personal reason for taking him down, that draws the two men into an explosive final showdown where only one will walk away alive.<br /><br />After Until Death, Van Damme appeared to be on a high, showing he could make the best straight to video films in the action market. While that was a far more drama oriented film, with The Shepherd he has returned to the high-kicking, no brainer action that first made him famous and has sadly produced his worst film since Derailed. It's nowhere near as bad as that film, but what I said still stands.<br /><br />A dull, predictable film, with very little in the way of any exciting action. What little there is mainly consists of some limp fight scenes, trying to look cool and trendy with some cheap slo-mo/sped up effects added to them that sadly instead make them look more desperate. Being a Mexican set film, director Isaac Florentine has tried to give the film a Robert Rodriguez/Desperado sort of feel, but this only adds to the desperation.<br /><br />VD gives a particularly uninspired performance and given he's never been a Robert De Niro sort of actor, that can't be good. As the villain, Lord shouldn't expect to leave the beeb anytime soon. He gets little dialogue at the beginning as he struggles to muster an American accent but gets mysteriously better towards the end. All the supporting cast are equally bland, and do nothing to raise the films spirits at all.<br /><br />This is one shepherd that's strayed right from the flock. *
#
# Sample prediction with positivity score 0.996307373046875:
# First off let me say, If you haven't enjoyed a Van Damme movie since bloodsport, you probably will not like this movie. Most of these movies may not have the best plots or best actors but I enjoy these kinds of movies for what they are. This movie is much better than any of the movies the other action guys (Segal and Dolph) have thought about putting out the past few years. Van Damme is good in the movie, the movie is only worth watching to Van Damme fans. It is not as good as Wake of Death (which i highly recommend to anyone of likes Van Damme) or In hell but, in my opinion it's worth watching. It has the same type of feel to it as Nowhere to Run. Good fun stuff!
# ```
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
# In order to improve this, what you could do is store the model in the image that
# backs each container.
# See [`Image.run_function`](/docs/guide/custom-container#run-a-modal-function-during-your-build-with-run_function-beta).
#
