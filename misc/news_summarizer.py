# # News article summarizer
#
# In this example we scrape news articles from the [New York Times'
# Science section](https://www.nytimes.com/section/science) and summarize them
# using Google's deep learning summarization model [Pegasus](https://ai.googleblog.com/2020/06/pegasus-state-of-art-model-for.html).
# We log the resulting summaries to the terminal, but you can do whatever you want with the
# summaries afterwards: saving to a CSV file, sending to Slack, etc.

import os
import re
from dataclasses import dataclass
from typing import List

import modal

app = modal.App(name="example-news-summarizer")

# ## Building Images and Downloading Pre-trained Model
#
# We start by defining our images. In Modal, each function can use a different
# image. This is powerful because you add only the dependencies you need for
# each function.

# The first image contains dependencies for running our model. We also download the
# pre-trained model into the image using the `from_pretrained` method.
# This caches the model so that we don't have to download it on every function call.
# The model will be saved at `/cache` when this function is called at image build time;
# subsequent calls of this function at runtime will then load the model from `/cache`.


def fetch_model(local_files_only: bool = False):
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer

    tokenizer = PegasusTokenizer.from_pretrained(
        "google/pegasus-xsum",
        cache_dir="/cache",
        local_files_only=local_files_only,
    )
    model = PegasusForConditionalGeneration.from_pretrained(
        "google/pegasus-xsum",
        cache_dir="/cache",
        local_files_only=local_files_only,
    )
    return model, tokenizer


deep_learning_image = (
    modal.Image.debian_slim()
    .pip_install("transformers==4.16.2", "torch", "sentencepiece")
    .run_function(fetch_model)
)

# Defining the scraping image is very similar. This image only contains the packages required
# to scrape the New York Times website, though; so it's much smaller.
scraping_image = modal.Image.debian_slim().pip_install(
    "requests", "beautifulsoup4", "lxml"
)


with scraping_image.imports():
    import requests
    from bs4 import BeautifulSoup


# ## Collect Data
#
# Collecting data happens in two stages: first a list of URL articles
# using the NYT API then scrape the NYT web page for each of those articles
# to collect article texts.


@dataclass
class NYArticle:
    title: str
    image_url: str = ""
    url: str = ""
    summary: str = ""
    text: str = ""


# In order to connect to the NYT API, you will need to sign up at [NYT Developer Portal](https://developer.nytimes.com/),
# create an App then grab an API key. Then head to Modal and create a [Secret](https://modal.com/docs/guide/secrets) called `nytimes`.
# Create an environment variable called `NYTIMES_API_KEY` with your API key.


@app.function(
    secrets=[modal.Secret.from_name("nytimes")],
    image=scraping_image,
)
def latest_science_stories(n_stories: int = 5) -> List[NYArticle]:
    # query api for latest science articles
    params = {
        "api-key": os.environ["NYTIMES_API_KEY"],
    }
    nyt_api_url = "https://api.nytimes.com/svc/topstories/v2/science.json"
    response = requests.get(nyt_api_url, params=params)

    # extract data from articles and return list of NYArticle objects
    results = response.json()
    reject_urls = {"null", "", None}
    articles = [
        NYArticle(
            title=u["title"],
            image_url=(u.get("multimedia")[0]["url"] if u.get("multimedia") else ""),
            url=u.get("url"),
        )
        for u in results["results"]
        if u.get("url") not in reject_urls
    ]

    # select only a handful of articles; this usually returns 25 articles
    articles = articles[:n_stories]
    print(f"Retrieved {len(articles)} from the NYT Top Stories API")
    return articles


# The NYT API only gives us article URLs but it doesn't include the article text. We'll get the article URLs
# from the API then scrape each URL for the article body. We'll be using
# [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) for that.


@app.function(image=scraping_image)
def scrape_nyc_article(url: str) -> str:
    print(f"Scraping article => {url}")

    # fetch article; simulate desktop browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "lxml")

    # get all text paragraphs & construct single string with article text
    article_text = ""
    article_section = soup.find_all(
        "div", {"class": re.compile(r"\bStoryBodyCompanionColumn\b")}
    )
    if article_section:
        paragraph_tags = article_section[0].find_all("p")
        article_text = " ".join([p.get_text() for p in paragraph_tags])

    # return article with scraped text
    return article_text


# Now the summarization function. We use `huggingface`'s Pegasus tokenizer and model implementation to
# generate a summary of the model. You can learn more about Pegasus does in the [HuggingFace
# documentation](https://huggingface.co/docs/transformers/model_doc/pegasus). Use `gpu="any"` to speed-up inference.


@app.function(
    image=deep_learning_image,
    gpu=False,
    memory=4096,
)
def summarize_article(text: str) -> str:
    print(f"Summarizing text with {len(text)} characters.")

    # `local_files_only` is set to `True` because we expect to read the model
    # files saved in the image.
    model, tokenizer = fetch_model(local_files_only=True)

    # summarize text
    batch = tokenizer(
        [text], truncation=True, padding="longest", return_tensors="pt"
    ).to("cpu")
    translated = model.generate(**batch)
    summary = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

    return summary


# ## Create a Scheduled Function
#
# Put everything together and schedule it to run every day. You can also use `modal.Cron` for a
# more advanced scheduling interface.


@app.function(schedule=modal.Period(days=1))
def trigger():
    articles = latest_science_stories.remote()

    # parallelize article scraping
    for i, text in enumerate(scrape_nyc_article.map([a.url for a in articles])):
        articles[i].text = text

    # parallelize summarization
    for i, summary in enumerate(
        summarize_article.map([a.text for a in articles if len(a.text) > 0])
    ):
        articles[i].summary = summary

    # show all summaries in the terminal
    for article in articles:
        print(f'Summary of "{article.title}" => {article.summary}')


# Create a new Modal scheduled function with:
#
# ```shell
# modal deploy --name news_summarizer news_summarizer.py
# ```

# You can also run this entire Modal app in debugging mode before.
# call it with `modal run news_summarizer.py`


@app.local_entrypoint()
def main():
    trigger.remote()


# And that's it. You will now generate deep learning summaries from the latest
# NYT Science articles every day.
