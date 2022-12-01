"""
Defines a serverless web API to expose trained models 
"""
from typing import Optional

import modal
from fastapi import FastAPI, Header
from pydantic import BaseModel

from . import config
from . import models
from .app import stub, volume

web_app = FastAPI()

if stub.is_inside():
    model_id = "sha256.4421D12E64E344F7F32E3A258E9A09CF9439A578787DFEABBAE656A0AB5E82E2"
    m = models.LLM()
    classifier = m.load(sha256_digest=model_id, model_registry_root=config.MODEL_STORE_DIR)
else:
    classifier = None


class ModelInput(BaseModel):
    text: str


class ModelOutput(BaseModel):
    spam: bool
    prob: float


@web_app.get("/api/v1/models")
async def handle_list_models(user_agent: Optional[str] = Header(None)):
    """
    Show details of actively serving models.
    """
    print(f"GET /     - received user_agent={user_agent}")
    return "Hello World"


@web_app.post("/api/v1/classify")
async def handle_classification(input_: ModelInput, user_agent: Optional[str] = Header(None)):
    """
    Classify a body of text as spam or ham.

    eg. 
    
    ```bash
    curl -X POST https://modal-labs-example-spam-detect-llm-fastapi-app.modal.run/api/v1/classify \ 
    -H 'Content-Type: application/json' \
    -d '{"text": "hello world"}'
    ```
    """
    print(f"POST /foo - received user_agent={user_agent}, {input_.text=}")
    # TODO(Jonathon):
    # - Load model serving config from a file ✔️
    # - Cache this information in a modal.Dict, with TTL.
    threshold = 0.5
    prob = classifier(input_.text)
    return ModelOutput(
        spam=(prob >= threshold),
        prob=prob,
    )


@stub.asgi(
    shared_volumes={config.VOLUME_DIR: volume},
)
def fastapi_app():
    return web_app


if __name__ == "__main__":
    stub.serve()
