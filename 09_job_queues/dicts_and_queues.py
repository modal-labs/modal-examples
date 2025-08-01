# ---
# mypy: ignore-errors
# ---

# # Use Modal Dicts and Queues together

# Modal Dicts and Queues store and communicate objects in distributed applications on Modal.

# To illustrate how Dicts and Queues can interact together in a simple distributed
# system, consider the following example program that crawls the web, starting
# from some initial page and traversing links to many sites in breadth-first order.

# The Modal Queue acts as a job queue, accepting new pages to crawl as they are discovered
# by the crawlers and doling them out to be crawled via [`.spawn`](https://modal.com/docs/reference/modal.Function#spawn).

# The Dict is used to coordinate termination once the maximum number of URLs to crawl is reached.

# Starting from Wikipedia, this spawns several dozen containers (auto-scaled on
# demand) and crawls about 100,000 URLs per minute.

import queue
import sys
from datetime import datetime

import modal

app = modal.App(
    "example-dicts-and-queues",
    image=modal.Image.debian_slim().pip_install(
        "requests~=2.32.4", "beautifulsoup4~=4.13.4"
    ),
)


def extract_links(url: str) -> list[str]:
    """Extract links from a given URL."""
    import urllib.parse

    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for link in soup.find_all("a"):
        links.append(urllib.parse.urljoin(url, link.get("href")))
    return links


@app.function()
def crawl_pages(q: modal.Queue, d: modal.Dict, urls: set[str]) -> None:
    for url in urls:
        if "stop" in d:
            return
        try:
            s = datetime.now()
            links = extract_links(url)
            print(f"Crawled: {url} in {datetime.now() - s}, with {len(links)} links")
            q.put_many(links)
        except Exception as exc:
            print(
                f"Failed to crawl: {url} with error {exc}, skipping...", file=sys.stderr
            )


@app.function()
def scrape(url: str, max_urls: int = 50_000):
    start_time = datetime.now()

    # Create ephemeral dicts and queues
    with modal.Dict.ephemeral() as d, modal.Queue.ephemeral() as q:
        # The dict is used to signal the scraping to stop
        # The queue contains the URLs that have been crawled

        # Initialize queue with a starting URL
        q.put(url)

        # Crawl until the queue is empty, or reaching some number of URLs
        visited = set()
        max_urls = min(max_urls, 50_000)
        while True:
            try:
                next_urls = q.get_many(2000, timeout=5)
            except queue.Empty:
                break
            new_urls = set(next_urls) - visited
            visited |= new_urls
            if len(visited) < max_urls:
                crawl_pages.spawn(q, d, new_urls)
            else:
                d["stop"] = True

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"Crawled {len(visited)} URLs in {elapsed:.2f} seconds")


@app.local_entrypoint()
def main(starting_url=None, max_urls: int = 10_000):
    starting_url = starting_url or "https://www.wikipedia.org/"
    scrape.remote(starting_url, max_urls=max_urls)
