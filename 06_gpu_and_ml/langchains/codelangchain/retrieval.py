"""Just as a constant function is _technically_ a polynomial, so too is injecting the same information every time _technically_ RAG."""
from common import COLOR

lcel_docs_url = "https://python.langchain.com/docs/expression_language/"


def retrieve_docs(url: str = lcel_docs_url, debug=False):
    from bs4 import BeautifulSoup as Soup
    from langchain_community.document_loaders.recursive_url_loader import (
        RecursiveUrlLoader,
    )

    print(
        f"{COLOR['HEADER']}📜: Retrieving documents from {url}{COLOR['ENDC']}"
    )
    loader = RecursiveUrlLoader(
        url=lcel_docs_url,
        max_depth=20 // (int(debug) + 1),  # retrieve fewer docs in debug mode
        extractor=lambda x: Soup(x, "html.parser").text,
    )
    docs = loader.load()

    # sort the list based on the URLs
    d_sorted = sorted(docs, key=lambda x: x.metadata["source"], reverse=True)

    # combine them all together
    concatenated_content = "\n\n\n --- \n\n\n".join(
        [
            "## " + doc.metadata["source"] + "\n\n" + doc.page_content.strip()
            for doc in d_sorted
        ]
    )

    print(
        f"{COLOR['HEADER']}📜: Retrieved {len(docs)} documents{COLOR['ENDC']}",
        f"{COLOR['GREEN']}{concatenated_content[:100].strip()}{COLOR['ENDC']}",
        sep="\n",
    )

    if debug:
        print(
            f"{COLOR['HEADER']}📜: Restricting to at most 30,000 characters{COLOR['ENDC']}"
        )
        concatenated_content = concatenated_content[:30_000]

    return concatenated_content
