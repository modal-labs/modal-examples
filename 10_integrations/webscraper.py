# ---
# deploy: true
# ---

# # A simple web scraper

# In this guide we'll introduce you to Modal by writing a simple web scraper.
# We'll explain the foundations of a Modal application step by step.

# ## Set up your first Modal app

# Modal Apps are orchestrated as Python scripts but can theoretically run
# anything you can run in a container. To get you started, make sure to install
# the latest `modal` Python package and set up an API token (the first two steps
# [here](https://modal.com/docs/guide)).

# ## Scrape links locally

# First, we create an empty Python file `webscraper.py`. This file will contain our
# application code. Let's write some basic Python code to fetch the contents of a
# web page and print the links (`href` attributes) it finds in the document:

# ```python
# import re
# import sys
# import urllib.request
#
#
# def get_links(url):
#     response = urllib.request.urlopen(url)
#     html = response.read().decode("utf8")
#     links = []
#     for match in re.finditer('href="(.*?)"', html):
#         links.append(match.group(1))
#     return links
#
#
# if __name__ == "__main__":
#     links = get_links(sys.argv[1])
#     print(links)
# ```

# Now obviously this is just pure standard library Python code, and you can run it
# on your machine:

# ```bash
# $ python webscraper.py http://example.com
# ['https://www.iana.org/domains/example']
# ```

# ## Run it on Modal

# To make the `get_links` function run on Modal instead of your local machine, all
# you need to do is

# - Import `modal`
# - Create a [`modal.App`](/docs/reference/modal.App) instance
# - Add an `@app.function()` annotation to your function
# - Replace the `if __name__ == "__main__":` block with a function decorated with
#   [`@app.local_entrypoint()`](/docs/reference/modal.App#local_entrypoint)
# - Call `get_links` using `get_links.remote`

# ```python
# import re
# import urllib.request
# import modal
#
# app = modal.App(name="example-webscraper")
#
#
# @app.function()
# def get_links(url):
#     response = urllib.request.urlopen(url)
#     html = response.read().decode("utf8")
#     links = []
#     for match in re.finditer('href="(.*?)"', html):
#         links.append(match.group(1))
#     return links
#
#
# @app.local_entrypoint()
# def main(url):
#     links = get_links.remote(url)
#     print(links)
# ```

# You can now run this with the Modal CLI, using `modal run` instead of `python`.
# This time, you'll see additional progress indicators while the script is
# running, something like:

# ```bash
# $ modal run webscraper.py --url http://example.com
# ✓ Initialized.
# ✓ Created objects.
# ['https://www.iana.org/domains/example']
# ✓ App completed.
# ```

# ## Add dependencies

# In the code above we make use of the Python standard library `urllib` library.
# This works great for static web pages, but many pages these days use javascript
# to dynamically load content, which wouldn't appear in the loaded html file.
# Let's use the [Playwright](https://playwright.dev/python/docs/intro) package to
# instead launch a headless Chromium browser which can interpret any javascript
# that might be on the page.

# We can pass [custom container images](/docs/guide/images) (defined using
# [`modal.Image`](/docs/reference/modal.Image)) to the `@app.function()`
# decorator. We'll make use of the `modal.Image.debian_slim` pre-bundled Image add
# the shell commands to install Playwright and its dependencies:

import modal

app = modal.App("example-webscraper")
playwright_image = modal.Image.debian_slim(python_version="3.10").run_commands(
    "apt-get update",
    "apt-get install -y software-properties-common",
    "apt-add-repository non-free",
    "apt-add-repository contrib",
    "pip install playwright==1.42.0",
    "playwright install-deps chromium",
    "playwright install chromium",
)

# Note that we don't have to install Playwright or Chromium on our development
# machine since this will all run in Modal. We can now modify our `get_links`
# function to make use of the new tools.


@app.function(image=playwright_image)
async def get_links(cur_url: str) -> list[str]:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(cur_url)
        links = await page.eval_on_selector_all(
            "a[href]", "elements => elements.map(element => element.href)"
        )
        await browser.close()

    print("Links", links)
    return list(set(links))


# Since Playwright has a nice async interface, we'll redeclare our `get_links`
# function as async (Modal works with both sync and async functions).

# The first time you run the function after making this change, you'll notice that
# the output first shows the progress of building the image you specified,
# after which your function runs like before. This image is then cached so that on
# subsequent runs of the function it will not be rebuilt as long as the image
# definition is the same.

# ## Scale out

# So far, our script only fetches the links for a single page. What if we want to
# scrape a large list of links in parallel?

# We can do this easily with Modal, because of some magic: the function we wrapped
# with the `@app.function()` decorator is no longer an ordinary function, but a
# Modal [Function](https://modal.com/docs/reference/modal.Function) object. This
# means it comes with a `map` property built in, that lets us run this function
# for all inputs in parallel, scaling up to as many workers as needed.

# Let's change our code to scrape all urls we feed to it in parallel:

# ```python
# @app.local_entrypoint()
# def main():
#     urls = ["http://modal.com", "http://github.com"]
#     for links in get_links.map(urls):
#         for link in links:
#             print(link)
# ```

# ## Deploy it and run it on a schedule

# Let's say we want to log the scraped links daily. We move the print loop into
# its own Modal function and annotate it with a `modal.Period(days=1)` schedule -
# indicating we want to run it once per day. Since the scheduled function will not
# run from our command line, we also add a hard-coded list of links to crawl for
# now. In a more realistic setting we could read this from a database or other
# accessible data source.

# ```python
# @app.function(schedule=modal.Period(days=1))
# def daily_scrape():
#     urls = ["http://modal.com", "http://github.com"]
#     for links in get_links.map(urls):
#         for link in links:
#             print(link)
# ```

# To deploy App permanently, run the command

# ```
# modal deploy webscraper.py
# ```

# Running this command deploys this function and then closes immediately. We can
# see the deployment and all of its runs, including the printed links, on the
# Modal [Apps page](https://modal.com/apps). Rerunning the script will redeploy
# the code with any changes you have made - overwriting an existing deploy with
# the same name ("example-webscraper" in this case).

# ## Add Secrets and integrate with other systems

# Instead of looking at the links in the run logs of our deployments, let's say we
# wanted to post them to a `#scraped-links` Slack channel. To do this, we can
# make use of the [Slack API](https://api.slack.com/) and the `slack-sdk`
# [PyPI package](https://pypi.org/project/slack-sdk/).

# The Slack SDK WebClient requires an API token to get access to our Slack
# Workspace, and since it's bad practice to hardcode credentials into application
# code we make use of Modal's **Secrets**. Secrets are snippets of data that will
# be injected as environment variables in the containers running your functions.

# The easiest way to create Secrets is to go to the
# [Secrets section of modal.com](https://modal.com/secrets). You can both create a
# free-form secret with any environment variables, or make use of presets for
# common services. We'll use the Slack preset and after filling in the necessary
# information we are presented with a snippet of code that can be used to post to
# Slack using our credentials, which looks something like:

import os

slack_sdk_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "slack-sdk"
)


@app.function(
    image=slack_sdk_image,
    secrets=[
        modal.Secret.from_name(
            "scraper-slack-secret", required_keys=["SLACK_BOT_TOKEN"]
        )
    ],
    retries=3,
)
def bot_token_msg(channel, message):
    import slack_sdk

    client = slack_sdk.WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    print(f"Posting {message} to #{channel}")
    client.chat_postMessage(channel=channel, text=message)


# Notice the `retries` in the `@app.function` decorator.
# That parameter adds automatic retries when Function calls fail
# due to temporary issues, like rate limits. Read more [here](https://modal.com/docs/guide/retries)

# Copy that code, then amend the `daily_scrape` function to call
# `bot_token_msg`. We also add a per-URL `limit` for good measure.


@app.function(schedule=modal.Period(days=1))
def daily_scrape(limit: int = 50):
    urls = ["http://modal.com", "http://github.com"]

    for links in get_links.map(urls):
        for link in links[:limit]:
            bot_token_msg.remote("scraped-links", link)


@app.local_entrypoint()
def main():
    urls = ["http://modal.com", "http://github.com"]
    for links in get_links.map(urls):
        for link in links:
            print(link)


# Note that we are freely making function calls across completely different
# container images, as if they were regular Python functions in the same program!

# We keep the `local_entrypoint` the same so that we can still `modal run`
# this script to test the scraping behavior without posting to Slack.

# ```bash
# modal run webscraper.py  # runs get_links.map via the local_entrypoint
# ```

# If we want to test the `daily_scrape` or `bot_token_msg` Functions themselves, we can do that too!
# We just add the name of the Function to the end of our `modal run` command:

# ```bash
# modal run webscraper.py::daily_scrape --limit 1  # quick test
# ```

# Now redeploy the script to overwrite the old deploy with our updated code, and
# you'll get a daily feed of scraped links in your Slack channel 🎉

# ```bash
# modal deploy webscraper.py
# ```

# ## Summary

# We have shown how you can use Modal to develop distributed Python data
# applications using custom containers. Through simple constructs we were able to
# add parallel execution. With the change of a single line of code were were able
# to go from experimental development code to a deployed application. We hope
# this overview gives you a glimpse of what you are able to build using Modal.
