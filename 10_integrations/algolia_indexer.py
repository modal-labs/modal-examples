# ---
# deploy: true
# env: {"MODAL_ENVIRONMENT": "main"}
# ---
# # Algolia docsearch crawler
#
# This tutorial shows you how to use Modal to run the [Algolia docsearch
# crawler](https://docsearch.algolia.com/docs/legacy/run-your-own/) to index your
# website and make it searchable. This is not just example code - we run the same
# code in production to power search on this page (`Ctrl+K` to try it out!).

# ## Basic setup
#
# Let's get the imports out of the way.

import json
import os
import subprocess

import modal

# Modal lets you [use and extend existing Docker images](/docs/guide/custom-container#use-an-existing-container-image-with-from_registry),
# as long as they have `python` and `pip` available. We'll use the official crawler image built by Algolia, with a small
# adjustment: since this image has `python` symlinked to `python3.6` and Modal is not compatible with Python 3.6, we
# install Python 3.11 and symlink that as the `python` executable instead.

algolia_image = modal.Image.from_registry(
    "algolia/docsearch-scraper:v1.16.0",
    add_python="3.11",
    setup_dockerfile_commands=["ENTRYPOINT []"],
)

app = modal.App("example-algolia-indexer")

# ## Configure the crawler
#
# Now, let's configure the crawler with the website we want to index, and which
# CSS selectors we want to scrape. Complete documentation for crawler configuration is available
# [here](https://docsearch.algolia.com/docs/legacy/config-file).

CONFIG = {
    "index_name": "modal_docs",
    "custom_settings": {
        "separatorsToIndex": "._",
        "synonyms": [["cls", "class"]],
    },
    "stop_urls": [
        "https://modal.com/docs/reference/modal.Stub",
    ],
    "start_urls": [
        {
            "url": "https://modal.com/docs/guide",
            "selectors_key": "default",
            "page_rank": 2,
        },
        {
            "url": "https://modal.com/docs/examples",
            "selectors_key": "examples",
            "page_rank": 1,
        },
        {
            "url": "https://modal.com/docs/reference",
            "selectors_key": "reference",
            "page_rank": 1,
        },
    ],
    "selectors": {
        "default": {
            "lvl0": {
                "selector": "header .navlink-active",
                "global": True,
            },
            "lvl1": "article h1",
            "lvl2": "article h2",
            "lvl3": "article h3",
            "text": "article p,article ol,article ul,article pre",
        },
        "examples": {
            "lvl0": {
                "selector": "header .navlink-active",
                "global": True,
            },
            "lvl1": "article h1",
            "text": "article p,article ol,article ul,article pre",
        },
        "reference": {
            "lvl0": {
                "selector": "//div[contains(@class, 'sidebar')]//a[contains(@class, 'active')]//preceding::a[contains(@class, 'header')][1]",
                "type": "xpath",
                "global": True,
                "default_value": "",
                "skip": {"when": {"value": ""}},
            },
            "lvl1": "article h1",
            "lvl2": "article h2",
            "lvl3": "article h3",
            "text": "article p,article ol,article ul,article pre",
        },
    },
}

# ## Create an API key
#
# If you don't already have one, sign up for an account on [Algolia](https://www.algolia.com/). Set up
# a project and create an API key with `write` access to your index, and with the ACL permissions
# `addObject`, `editSettings` and `deleteIndex`. Now, create a Secret on the Modal [Secrets](https://modal.com/secrets)
# page with the `API_KEY` and `APPLICATION_ID` you just created. You can name this anything you want,
# but we named it `algolia-secret` and so that's what the code below expects.

# ## The actual function
#
# We want to trigger our crawler from our CI/CD pipeline, so we're serving it as a
# [web endpoint](/docs/guide/webhooks#web_endpoint) that can be triggered by a `GET` request during deploy.
# You could also consider running the crawler on a [schedule](/docs/guide/cron).
#
# The Algolia crawler is written for Python 3.6 and needs to run in the `pipenv` created for it,
# so we're invoking it using a subprocess.


@app.function(
    image=algolia_image,
    secrets=[modal.Secret.from_name("algolia-secret")],
)
def crawl():
    # Installed with a 3.6 venv; Python 3.6 is unsupported by Modal, so use a subprocess instead.
    subprocess.run(
        ["pipenv", "run", "python", "-m", "src.index"],
        env={**os.environ, "CONFIG": json.dumps(CONFIG)},
    )


# We want to be able to trigger this function through a webhook.


@app.function(image=modal.Image.debian_slim().pip_install("fastapi[standard]"))
@modal.web_endpoint()
def crawl_webhook():
    crawl.remote()
    return "Finished indexing docs"


# ## Deploy the indexer

# That's all the code we need! To deploy your application, run

# ```shell
# modal deploy algolia_indexer.py
# ```

# If successful, this will print a URL for your new webhook, that you can hit using
# `curl` or a browser. Logs from webhook invocations can be found from the [apps](/apps)
# page.

# The indexed contents can be found at https://www.algolia.com/apps/APP_ID/explorer/browse/, for your
# APP_ID. Once you're happy with the results, you can [set up the `docsearch` package with your
# website](https://docsearch.algolia.com/docs/docsearch-v3/), and create a search component that uses this index.

# ## Entrypoint for development

# To make it easier to test this, we also have an entrypoint for when you run
# `modal run algolia_indexer.py`


@app.local_entrypoint()
def run():
    crawl.remote()
