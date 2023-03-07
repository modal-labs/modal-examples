# ---
# cmd: ["modal", "run", "misc.webscraper::scrape"]
# ---
import os

import modal

stub = modal.Stub("example-linkscraper")


playwright_image = modal.Image.debian_slim().run_commands(
    "apt-get install -y software-properties-common",
    "apt-add-repository non-free",
    "apt-add-repository contrib",
    "apt-get update",
    "pip install playwright==1.20.0",
    "playwright install-deps chromium",
    "playwright install chromium",
)


@stub.function(image=playwright_image)
async def get_links(url: str):
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        links = await page.eval_on_selector_all(
            "a[href]", "elements => elements.map(element => element.href)"
        )
        await browser.close()

    return links


slack_sdk_image = modal.Image.debian_slim().pip_install("slack-sdk")


@stub.function(
    image=slack_sdk_image, secret=modal.Secret.from_name("scraper-slack-secret")
)
def bot_token_msg(channel, message):
    import slack_sdk

    client = slack_sdk.WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    client.chat_postMessage(channel=channel, text=message)


@stub.function
def scrape():
    links_of_interest = ["http://modal.com"]

    for links in get_links.map(links_of_interest):
        for link in links:
            bot_token_msg.call("scraped-links", link)


@stub.function(schedule=modal.Period(days=1))
def daily_scrape():
    scrape.call()


if __name__ == "__main__":
    with stub.run():
        scrape.call()
