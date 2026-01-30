import modal

# Optimized Image: Caches the browser so it's ready in <1s
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("playwright==1.42.0")
    .run_commands("playwright install chromium")
    .run_commands("playwright install-deps chromium")
)

app = modal.App("misc-dist-crawler")

# Persistent storage for the crawl results
visited_dict = modal.Dict.from_name("crawler-visited-metadata", create_if_missing=True)

@app.function(
    image=image, 
    concurrency_limit=50, 
    timeout=300, # Increased for slow pages
    retries=1
)
async def crawl_page(url: str):
    from playwright.async_api import async_playwright
    
    # Check-then-set logic to minimize redundant crawls
    if await visited_dict.contains.aio(url):
        return []

    print(f"🕵️ Crawling: {url}")
    links = []

    async with async_playwright() as p:
        # Args optimized for serverless/headless environments
        browser = await p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = await browser.new_page()
        
        try:
            # wait_until="domcontentloaded" is faster than "networkidle"
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            
            # Extract absolute URLs using browser-side JS
            raw_links = await page.eval_on_selector_all(
                "a[href]", "elements => elements.map(e => e.href)"
            )
            
            # Clean up: only keep valid web links
            links = [l.split('#')[0] for l in raw_links if l.startswith("http")]
            
            # Record success
            await visited_dict.put.aio(url, True)
        except Exception as e:
            print(f"⚠️ Error crawling {url}: {str(e)[:50]}...")
        finally:
            await browser.close()
            
    return list(set(links)) # Deduplicate locally before returning

@app.local_entrypoint()
def main(seed_url: str = "https://modal.com", depth: int = 2):
    print(f"🚀 Starting Crawler at {seed_url} (Depth: {depth})")
    
    queue = [seed_url]
    for d in range(depth):
        print(f"--- Level {d}: Processing {len(queue)} URLs ---")
        
        responses = []
        # .map() fans out to 50+ parallel containers
        for found_links in crawl_page.map(queue):
            responses.extend(found_links)
        
        # Prepare the next layer: links we haven't visited yet
        unique_links = set(responses)
        queue = [l for l in unique_links if l not in visited_dict]
        
        if not queue:
            break

    print(f"✅ Finished. Total pages visited: {len(visited_dict)}")
