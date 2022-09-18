# ---
# integration-test: false
# args: ["http://example.com"]
# ---
import asyncio
import sys
from dataclasses import dataclass

from modal.aio import AioImage, AioDict, AioQueue, AioStub

stub = AioStub()

image = AioImage.debian_slim().run_commands(
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


@dataclass
class Crawler:
    q: AioQueue
    visited: AioDict
    pending: AioDict
    max_depth: int
    root: str


async def get_links(cur_url: str):
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
    return links


@stub.function(image=image)
async def do_crawl_url(crawler: Crawler, cur_url: str, cur_depth: int):
    print("Crawling", cur_url, cur_depth)
    try:
        for nxt_url in await get_links(cur_url):
            await try_crawl_url(crawler, nxt_url, cur_depth + 1)
    finally:
        await crawler.pending.pop((cur_url, cur_depth))


async def try_crawl_url(crawler: Crawler, cur_url: str, cur_depth: int = 0):
    if not cur_url.startswith(crawler.root):
        return

    if (
        await crawler.visited.contains(cur_url)
        and await crawler.visited.get(cur_url) <= cur_depth
    ):
        return

    await crawler.visited.put(cur_url, cur_depth)
    await crawler.q.put((cur_url, cur_depth))

    print("Found", cur_url, cur_depth)

    if cur_depth >= crawler.max_depth:
        return

    await crawler.pending.put((cur_url, cur_depth), True)
    await do_crawl_url.submit(crawler, cur_url, cur_depth)


@stub.function
async def main(root):
    crawler = Crawler(
        q=await AioQueue().create(),
        visited=await AioDict().create(),
        pending=await AioDict().create(),
        max_depth=3,
        root=root,
    )

    await try_crawl_url(crawler, root)
    while await crawler.pending.len() > 0:
        await asyncio.sleep(0.1)

    print("Finished", await crawler.visited.len())


async def _run(root):
    async with stub.run():
        await main(root)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Usage: examples.webcrawler [ROOT_URL]")
        exit(1)
    root = sys.argv[1]
    asyncio.run(_run(root))
