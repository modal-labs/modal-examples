# ---
# pytest: false
# ---
#
# # Anthropic Claude Batch Inference with Structured Output
#
# This example shows how to run concurrent batch inference using
# [Anthropic Claude](https://docs.anthropic.com/en/docs/about-claude/models/overview)
# and [Instructor](https://python.useinstructor.com/) for structured extraction,
# all on Modal.
#
# Two features are demonstrated:
#
# 1. **Batch CLI** — a `@app.local_entrypoint` that fans out extraction requests
#    across Modal workers using `.map()`.
# 2. **REST API** — a FastAPI endpoint (`POST /extract`) protected by an
#    `X-API-Key` header that accepts a list of texts and returns structured results.
#
# ## Setup
#
# You need two Modal Secrets:
#
# - `anthropic-secret` — must contain `ANTHROPIC_API_KEY`
# - `batch-api-key` — must contain `API_KEY` (the key callers send as `X-API-Key`)
#
# Create them in the [Modal dashboard](https://modal.com/secrets) or with the CLI:
#
# ```bash
# modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...
# modal secret create batch-api-key API_KEY=your-gateway-key
# ```
#
# ## Run the batch CLI
#
# ```bash
# modal run anthropic_batch_inference.py
# ```
#
# ## Deploy the API
#
# ```bash
# modal deploy anthropic_batch_inference.py
# ```
#
# Then call the deployed endpoint:
#
# ```bash
# curl -X POST https://<your-workspace>--example-anthropic-batch-inference-fastapi-app.modal.run/extract \
#   -H "X-API-Key: your-gateway-key" \
#   -H "Content-Type: application/json" \
#   -d '{"texts": ["Marie Curie won two Nobel Prizes.", "Elon Musk founded SpaceX in 2002."]}'
# ```

import os

import modal
from pydantic import BaseModel, Field

# ## App and image
#
# We install `anthropic` for the Claude API, `instructor` for structured
# extraction, and `fastapi[standard]` for the HTTP gateway.

app = modal.App("example-anthropic-batch-inference")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "anthropic>=0.50.0",
    "instructor>=1.7.0",
    "fastapi[standard]",
    "pydantic>=2.0",
)

# ## Pydantic models for structured output
#
# Instructor uses these to coerce Claude's response into typed Python objects.

with image.imports():
    import anthropic
    import instructor

# ## Entity extraction models


class Entity(BaseModel):
    text: str = Field(description="The surface form of the entity as it appears in the text")
    label: str = Field(description="Entity type, e.g. PERSON, ORG, GPE, DATE, EVENT")


class EntityExtractionResult(BaseModel):
    entities: list[Entity] = Field(description="All named entities found in the text")
    summary: str = Field(description="One-sentence summary of the text")


# ## Secrets
#
# Defining these at module level lets both the batch function and the API
# share the same secret references without repetition.

ANTHROPIC_SECRET = modal.Secret.from_name(
    "anthropic-secret", required_keys=["ANTHROPIC_API_KEY"]
)
API_KEY_SECRET = modal.Secret.from_name(
    "batch-api-key", required_keys=["API_KEY"]
)

# ## Extraction function
#
# Each worker calls Claude once.  Instructor handles the structured-output
# loop automatically so we always get back a validated `EntityExtractionResult`.


@app.function(image=image, secrets=[ANTHROPIC_SECRET])
def extract_entities(text: str) -> dict:
    client = instructor.from_anthropic(anthropic.Anthropic())
    result: EntityExtractionResult = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract all named entities from the following text "
                    "and provide a one-sentence summary.\n\n"
                    f"Text: {text}"
                ),
            }
        ],
        response_model=EntityExtractionResult,
    )
    return result.model_dump()


# ## Sample passages for the CLI demo

SAMPLE_PASSAGES = [
    "Marie Curie was a Polish-French physicist who won two Nobel Prizes, "
    "one in Physics in 1903 and one in Chemistry in 1911.",
    "Elon Musk founded SpaceX in Hawthorne, California in 2002 with the "
    "goal of reducing space transportation costs.",
    "The Treaty of Versailles, signed in June 1919, officially ended "
    "World War I between Germany and the Allied Powers.",
    "Ada Lovelace, daughter of poet Lord Byron, is often credited as the "
    "first computer programmer for her work on Charles Babbage's Analytical Engine.",
]

# ## Local entrypoint
#
# Running `modal run anthropic_batch_inference.py` fans the passages out to
# Modal workers in parallel via `.map()` and prints each result.


@app.local_entrypoint()
def main():
    print(f"Extracting entities from {len(SAMPLE_PASSAGES)} passages in parallel...\n")
    for passage, result in zip(
        SAMPLE_PASSAGES,
        extract_entities.map(SAMPLE_PASSAGES),
    ):
        print(f"Passage : {passage[:60]}...")
        print(f"Summary : {result['summary']}")
        print(f"Entities: {[e['text'] for e in result['entities']]}")
        print()


# ## FastAPI gateway
#
# `@modal.asgi_app()` exposes the FastAPI app as a Modal web endpoint.
# The `X-API-Key` header is verified against the `API_KEY` secret before
# any extraction work is performed.


@app.function(image=image, secrets=[ANTHROPIC_SECRET, API_KEY_SECRET])
@modal.asgi_app()
def fastapi_app():
    from fastapi import Depends, FastAPI, HTTPException, Security
    from fastapi.security import APIKeyHeader
    from pydantic import BaseModel as PydanticBaseModel

    web = FastAPI(
        title="Anthropic Batch Inference API",
        description="Parallel named-entity extraction with Claude + Instructor",
    )

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def verify_key(key: str = Security(api_key_header)):
        expected = os.environ.get("API_KEY", "")
        if not key or key != expected:
            raise HTTPException(status_code=403, detail="Invalid or missing API key")
        return key

    class BatchRequest(PydanticBaseModel):
        texts: list[str] = Field(
            description="List of text passages to extract entities from",
            min_length=1,
        )

    class BatchResponse(PydanticBaseModel):
        results: list[dict]
        count: int

    @web.post("/extract", response_model=BatchResponse, dependencies=[Depends(verify_key)])
    def batch_extract(request: BatchRequest):
        """Extract named entities from a batch of texts in parallel."""
        results = list(extract_entities.map(request.texts))
        return BatchResponse(results=results, count=len(results))

    @web.get("/health")
    def health():
        return {"status": "ok"}

    return web
