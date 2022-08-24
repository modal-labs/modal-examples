# ---
# integration-test: false
# output-directory: "/tmp/screenshots"
# ---
# # Screenshot with headless Chromium

# In this example, we use Modal functions and the `playwright` package to screenshot websites from a list of urls in parallel.
# Please also see our [introductory guide](/docs/guide/web-scraper) for another example of a web-scraper, with more in-depth examples.
# Basic setup first:

import asyncio
import os
import random
import traceback

import modal

stub = modal.Stub()

# ## Defining a custom image
#
# We need an image with the `playwright` Python package as well as its `chromium` plugin pre-installed.
# This requires intalling a few Debian packages, as well as setting up a new Debian repository.
# Modal lets you run arbitrary commands, just like in Docker:


image = modal.DebianSlim().run_commands(
    [
        "apt-get install -y software-properties-common",
        "apt-add-repository non-free",
        "apt-add-repository contrib",
        "apt-get update",
        "pip install playwright==1.20.0",
        "playwright install-deps chromium",
        "playwright install chromium",
    ],
)

# ## Defining the screenshot function
#
# Next, the scraping function which runs headless Chromium, goes to a website, and takes a screenshot.
# Note that this is not a Modal function, but it is invoked by a Modal function!


async def screenshot_get(url):
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        await page.screenshot(path="screenshot.png")
        await browser.close()
        data = open("screenshot.png", "rb").read()
        print("Screenshot of size %d bytes" % len(data))
        return data


# ## Defining a Modal function
#
# The actual modal.function. We want to handle failures ourselves, so the scraping code is in a try/except.
# `modal.RateLimit` applies a global rate limiter on all function calls, so we're guaranteed that this function
# won't execute more than two times a second.


@stub.function(image=image, rate_limit=modal.RateLimit(per_second=2))
async def screenshot(url):
    print("Fetching url", url)
    try:
        return await asyncio.wait_for(screenshot_get(url), 20.0)
    except Exception:
        traceback.print_exc()
        return None


# ## Entrypoint code
#
# Let's kick it off by reading a bunch of URLs from a txt file and scrape some of those.

OUTPUT_DIR = "/tmp/screenshots"

if __name__ == "__main__":
    urls_fn = os.path.join(os.path.dirname(__file__), "urls.txt")
    urls = ["http://" + line.strip() for line in open(urls_fn)]

    sampled_urls = [random.choice(urls) for i in range(3)]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with stub.run():
        for i, data in enumerate(screenshot.map(sampled_urls)):
            if data is not None:
                with open(os.path.join(OUTPUT_DIR, "%06d.png" % i), "wb") as f:
                    f.write(data)
