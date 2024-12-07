# # Web Scraping on Modal

# This example shows how you can scrape links from a website and post them to a Slack channel using Modal.

import os

import modal

app = modal.App("example-linkscraper")


playwright_image = modal.Image.debian_slim(
    python_version="3.10"
).run_commands(  # Doesn't work with 3.11 yet
    "apt-get update",
    "apt-get install -y software-properties-common",
    "apt-add-repository non-free",
    "apt-add-repository contrib",
    "pip install playwright==1.42.0",
    "playwright install-deps chromium",
    "playwright install chromium",
)


@app.function(image=playwright_image)
async def get_links(url: str) -> set[str]:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        links = await page.eval_on_selector_all(
            "a[href]", "elements => elements.map(element => element.href)"
        )
        await browser.close()

    return set(links)


slack_sdk_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "slack-sdk==3.27.1"
)


@app.function(
    image=slack_sdk_image,
    secrets=[
        modal.Secret.from_name(
            "scraper-slack-secret", required_keys=["SLACK_BOT_TOKEN"]
        )
    ],
)
def bot_token_msg(channel, message):
    import slack_sdk
    from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler

    client = slack_sdk.WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    rate_limit_handler = RateLimitErrorRetryHandler(max_retry_count=3)
    client.retry_handlers.append(rate_limit_handler)

    print(f"Posting {message} to #{channel}")
    client.chat_postMessage(channel=channel, text=message)


@app.function()
def scrape():
    links_of_interest = ["http://modal.com"]

    for links in get_links.map(links_of_interest):
        for link in links:
            bot_token_msg.remote("scraped-links", link)


@app.function(schedule=modal.Period(days=1))
def daily_scrape():
    scrape.remote()


@app.local_entrypoint()
def run():
    scrape.remote()
