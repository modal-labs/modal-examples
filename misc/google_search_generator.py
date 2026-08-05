# ---
# runtimes: ["runc", "gvisor"]
# ---
#
# # Use a generator to fetch search results
#
# This is a simple example which
#
# 1. Installs a custom Python package.
# 2. Uses a _generator_ to return results back to the launcher process.

import modal

# We build a custom image by adding the `google` package to the base image.
app = modal.App(
    "example-google-search-generator",
    image=modal.Image.debian_slim().pip_install("google"),
)

# Next, let's define a _generator_ function that uses our custom image.


@app.function()
def scrape(query):
    from googlesearch import search

    for url in search(query.encode(), stop=100):
        yield url


# Finally, let's launch it from the command line with `modal run`:


@app.local_entrypoint()
def main(query: str = "modal"):
    for url in scrape.remote_gen(query):
        print(url)
