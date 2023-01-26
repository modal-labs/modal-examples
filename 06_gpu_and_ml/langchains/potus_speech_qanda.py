# # Question-answering with LangChain
#
# In this example we create a large-language-model (LLM) powered question answering
# web endpoint and CLI. Only a single document is used as the knowledge-base of the application,
# the 2022 USA State of the Union address by President Joe Biden, but this same application structure
# could be extended to do question-answering over all State of the Union speeches, or other large text corpuses.
#
# It's the [LangChain](https://github.com/hwchase17/langchain) library that makes this all so easy. This demo is only around 100 lines of code!

from pathlib import Path

import modal

image = modal.Image.debian_slim().pip_install(
    "beautifulsoup4~=4.11.1",
    "httpx~=0.23.3",
    "faiss-cpu~=1.7.3",
    "langchain~=0.0.7",
    "lxml~=4.9.2",
    "openai~=0.26.3",
)
stub = modal.Stub(name="example-langchain-qanda", image=image, secrets=[modal.Secret.from_name("openai-secret")])

# Terminal codes for pretty-printing.
BOLD = "\033[1m"
END = "\033[0m"
# Filepaths for caching data on disk.
SPEECH_FILE_PATH = Path("state-of-the-union.txt")
# An embedding index that's relatively expensive to computer, so its cached in this global.
docsearch = None

# ## Downloading the speech from whitehouse.gov
#
# It's super easy to scrape the transcipt of Biden's speech using `httpx` and `BeautifulSoup`.
# This speech is just one document, and it's relatively short, but it's enough to demonstrate
# the question-answering capability of the LLM chain and it's clear that further documents could
# scraped and added to expand the demo, but let's keep things minimal.


def scrape_state_of_the_union() -> str:
    import httpx
    from bs4 import BeautifulSoup

    url = "https://www.whitehouse.gov/state-of-the-union-2022/"

    # fetch article; simulate desktop browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9"
    }
    response = httpx.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "lxml")

    # get all text paragraphs & construct single string with article text
    speech_text = ""
    speech_section = soup.find_all("div", {"class": "sotu-annotations__content"})
    if speech_section:
        paragraph_tags = speech_section[0].find_all("p")
        speech_text = "".join([p.get_text() for p in paragraph_tags])

    # return article with scraped text
    return speech_text.replace("\t", "")


# ## Constructing the Q&A chain
#
# At a high-level, this LLM chain will be able to answer questions asked about Biden's speech and provide
# references to which parts of the speech contain the evidence for given answers.
#
# The chain combines text-embedding index over parts of Biden's speech with OpenAI's GPT LLM.
# The index is used to select the most likely relevant parts of the speech given the question, and these
# are used to build a specialized prompt for the language model.
#
# For more information on the this, see [langchain.readthedocs.io/en/latest/use_cases/question_answering](https://langchain.readthedocs.io/en/latest/use_cases/question_answering).


def retrieve_sources(sources_refs: str, texts: list[str]) -> list[str]:
    """Map back from the references given by the LLM's output to the original text parts."""
    target_indices = [int(r.replace("-pl", "")) for r in sources_refs.split(",")]
    return [texts[i] for i in target_indices]


def qanda_langchain(query: str) -> tuple[str, list[str]]:
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.llms import OpenAI
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores.faiss import FAISS

    if SPEECH_FILE_PATH.exists():
        state_of_the_union = SPEECH_FILE_PATH.read_text()
    else:
        print("scraping the State of the Union speech")
        state_of_the_union = scrape_state_of_the_union()
        SPEECH_FILE_PATH.write_text(state_of_the_union)

    # We cannot send the entire speech to the model, as OpenAI's model has maximum limit on input tokens.
    # So we split up the speech into smaller chunks.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    print("splitting speech into text chunks")
    texts = text_splitter.split_text(state_of_the_union)

    # Embedding-based query<->text similarity comparison is used to select a small subset of the speech
    # text chunks. Generating the `docsearch` index is too slow to re-run on every request, so we do rudimentary caching
    # using a global variable.
    global docsearch

    if not docsearch:
        print("generating docsearch indexer")
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))])

    print("selecting text parts by similarity to query")
    docs = docsearch.similarity_search(query)

    chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")
    print("running query against Q&A chain")
    result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    answer, sources_refs = result["output_text"].split("SOURCES: ")
    sources = retrieve_sources(sources_refs, texts)
    return answer.strip(), sources


# ## Modal Functions
#
# With our application's functionality implemented we can hook it into Modal.
# As said about, we're implementing a web endpoint, `web`, and a CLI command, `cli`.


@stub.webhook(method="GET")
def web(query: str, show_sources: bool = False):
    answer, sources = qanda_langchain(query)
    if show_sources:
        return {
            "answer": answer,
            "sources": sources,
        }
    else:
        return {
            "answer": answer,
        }


@stub.function
def cli(query: str, show_sources: bool = False):
    answer, sources = qanda_langchain(query)

    print(f"🤖 {BOLD}ANSWER:{END}")
    print(answer)
    if show_sources:
        print(f"📃 {BOLD}SOURCES:{END}")
        for text in sources:
            print(text)
            print("----")


# ## Test run the CLI
#
# ```bash
# modal run potus_speech_qanda.py --query "What did the president say about Justice Breyer"
# 🤖 ANSWER:
# The president thanked Justice Breyer for his service and mentioned his legacy of excellence. He also nominated Ketanji Brown Jackson to continue in Justice Breyer's legacy.
# ```
#
# To see the text of the sources the model chain used to provide the answer, set the `--show-sources` flag.
#
# ```bash
# modal run potus_speech_qanda.py --query "How many oil barrels were released from reserves" --show-sources=True
# ```
#
# ## Test run the web endpoint
#
# Modal makes it trivially easy to easy langchain chains to the web. We can test drive this app's web endpoint
# by running `modal serve potus_speech_qanda.py` and then hitting the endpoint with `curl`:
#
# ```bash
# curl --get \
#   --data-urlencode "query=What did the president say about Justice Breyer" \
#   https://modal-labs--example-langchain-qanda-web.modal.run
# {
#   "answer": "The president thanked Justice Breyer for his service and mentioned his legacy of excellence. He also nominated Ketanji Brown Jackson to continue in Justice Breyer's legacy."
# }
# ```
