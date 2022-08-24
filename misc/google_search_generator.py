# # Use a generator to fetch search results
#
# This is a simple example which
#
# 1. Installs a custom Python package.
# 2. Uses a _generator_ to return results back to the launcher process.

import sys

import modal

# We build a custom image by adding the `google` package to the base image.
stub = modal.Stub(image=modal.DebianSlim().pip_install(["google"]))

# Next, let's define a _generator_ function that uses our custom image.


@stub.generator
def scrape(query):
    from googlesearch import search

    for url in search(query.encode(), stop=100):
        yield url


# Finally, let's launch it from the command line by searching for the first argument:

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "modal"
    with stub.run():
        for url in scrape(query):
            print(url)
