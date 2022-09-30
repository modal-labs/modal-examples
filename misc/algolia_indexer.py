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

# Modal lets you [use and extend existing Docker images](/docs/guide/custom-container#using-existing-docker-hub-images),
# as long as they have `python` and `pip` available. We'll use the official crawler image built by Algolia, with a small
# adjustment: since this image has `python` symlinked to `python3.6` and Modal is not compatible with Python 3.6, we
# update the symlink to point to `python3.7` instead (which convenentiently also happens to be installed).

algolia_image = modal.Image.from_dockerhub(
    tag="algolia/docsearch-scraper",
    setup_commands=["ln -sfn /usr/bin/python3.7 /usr/bin/python"],
)

stub = modal.Stub("algolia-indexer", image=algolia_image)

# ## Configure the crawler
#
# Now, let's configure the crawler with the website we want to index, and which
# CSS selectors we want to scrape. Complete documentation for crawler configuration is available
# [here](https://docsearch.algolia.com/docs/legacy/config-file).

CONFIG = {
    "index_name": "modal_docs",
    "start_urls": ["https://modal.com/docs"],
    "selectors": {
        "lvl0": "article h1",
        "lvl1": "article h1",
        "lvl2": "article h2",
        "lvl3": "article h3",
        "text": "article p,article ol,article ul",
    },
}

# ## Create an API key
#
# If you don't already have one, sign up for an account on [Algolia](https://www.algolia.com/). Set up
# a project and create an API key with `write` access to your index, and with the ACL permissions
# `addObject`, `editSettings` and `deleteIndex`. Now, create a secret on the Modal [Secrets](/secrets)
# page with the `API_KEY` and `APPLICATION_ID` you just created. You can name this anything you want,
# we named it `algolia-secret`.

# ## The actual function
#
# We want to trigger our crawler from our CI/CD pipeline, so we're serving it as a
# [webhook](/docs/guide/webhooks#webhook) that can be triggered by a `GET` request during deploy.
# You could also consider running the crawler on a [schedule](/docs/guide/cron).
#
# The Algolia crawler is written for Python 3.6 and needs to run in the `pipenv` created for it,
# so we're invoking it using a subprocess.


@stub.webhook(secrets=[modal.ref("algolia-secret")])
def crawl():
    # Installed with a 3.6 venv; Python 3.6 is unsupported by Modal, so use a subprocess instead.
    subprocess.run(
        ["pipenv", "run", "python", "-m", "src.index"],
        env={**os.environ, "CONFIG": json.dumps(CONFIG)},
    )
    return "Crawl completed"


# ## Deploy the Slackbot
#
# That's all the code we need! To deploy your application, run
#
# ```shell
# modal app deploy algolia_indexer.py
# ```
#
# If successful, this will print a URL for your new webhook, that you can hit using
# `curl` or a browser. Logs from webhook invocations can be found from the [apps](/apps)
# page.
#
# The indexed contents can be found at https://www.algolia.com/apps/APP_ID/explorer/browse/, for your
# APP_ID. Once you're happy with the results, you can [set up the `docsearch` package with your
# website](https://docsearch.algolia.com/docs/docsearch-v3/), and create a search component that uses this index.
