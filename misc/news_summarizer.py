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

# ## Building Images and Downloading Pre-trained Model
#
# We start by defining our images. In Modal, each function can use a different
# image. This is powerful because you add only the dependencies you need for
# each function.

stub = modal.Stub("example-news-summarizer")
MODEL_NAME = "google/pegasus-xsum"
CACHE_DIR = "/cache"

# The first image contains dependencies for running our model. We also download the
# pre-trained model into the image using the `huggingface` API. This caches the model so that
# we don't have to download it on every function call.
stub["deep_learning_image"] = modal.Image.debian_slim().pip_install(["transformers==4.16.2", "torch", "sentencepiece"])

# Defining the scraping image is very similar. This image only contains the packages required
# to scrape the New York Times website, though; so it's much smaller.
stub["scraping_image"] = modal.Image.debian_slim().pip_install(["requests", "beautifulsoup4", "lxml"])

volume = modal.SharedVolume().persist("pegasus-modal-vol")

# We will also instantiate the model and tokenizer globally so it’s available for all functions that use this image.
if stub.is_inside(stub["deep_learning_image"]):
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer

    TOKENIZER = PegasusTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    MODEL = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)


if stub.is_inside(stub["scraping_image"]):
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
# create an Stub then grab an API key. Then head to Modal and create a [Secret](https://modal.com/docs/guide/secrets) called `nytimes`.
# Create an environment variable called `NYTIMES_API_KEY` with your API key.


@stub.function(secret=modal.Secret.from_name("nytimes"), image=stub["scraping_image"])
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
            image_url=u.get("multimedia")[0]["url"] if u.get("multimedia") else "",
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


@stub.function(image=stub["scraping_image"])
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
    article_section = soup.find_all("div", {"class": re.compile(r"\bStoryBodyCompanionColumn\b")})
    if article_section:
        paragraph_tags = article_section[0].find_all("p")
        article_text = " ".join([p.get_text() for p in paragraph_tags])

    # return article with scraped text
    return article_text


# Now the summarization function. We use `huggingface`'s Pegasus tokenizer and model implementation to
# generate a summary of the model. You can learn more about Pegasus does in the [HuggingFace
# documentation](https://huggingface.co/docs/transformers/model_doc/pegasus). Use `gpu=True` to speed-up inference.


@stub.function(
    image=stub["deep_learning_image"],
    gpu=False,
    shared_volumes={CACHE_DIR: volume},
    memory=4096,
)
def summarize_article(text: str) -> str:

    print(f"Summarizing text with {len(text)} characters.")

    # summarize text
    batch = TOKENIZER([text], truncation=True, padding="longest", return_tensors="pt").to("cpu")
    translated = MODEL.generate(**batch)
    summary = TOKENIZER.batch_decode(translated, skip_special_tokens=True)[0]

    return summary


# ## Create a Scheduled Function
#
# Put everything together and schedule it to run every day. You can also use `modal.Cron` for a
# more advanced scheduling interface.


@stub.function(schedule=modal.Period(days=1))
def trigger():
    articles = latest_science_stories.call()

    # parallelize article scraping
    for i, text in enumerate(scrape_nyc_article.map([a.url for a in articles])):
        articles[i].text = text

    # parallelize summarization
    for i, summary in enumerate(summarize_article.map([a.text for a in articles if len(a.text) > 0])):
        articles[i].summary = summary

    # show all summaries in the terminal
    for article in articles:
        print(f'Summary of "{article.title}" => {article.summary}')


# Create a new Modal scheduled function with:
#
# ```shell
# modal app deploy --name news_summarizer news_summarizer.py::stub
# ```

# You can also run this entire Modal app in debugging mode before.
# call it with regular python as `python news_summarizer.py`
if __name__ == "__main__":
    with stub.run():
        trigger.call()

# And that's it. You will now generate deep learning summaries from the latest
# NYT Science articles every day.
