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
    model_id = "sha256.4D4CA273952449C9D20E837F4425DC012C1BABF9AFD4D8E118BB50A596C72B87"
    m = models.LLM()
    classifier = m.load(sha256_digest=model_id, model_registry_root=config.MODEL_STORE_DIR)
else:
    classifier = None


class ModelInput(BaseModel):
    text: str


class ModelOutput(BaseModel):
    spam: bool
    score: float


@web_app.get("/api/v1/models")
async def handle_list_models(user_agent: Optional[str] = Header(None)):
    """
    Show details of actively serving models.
    """
    print(f"GET /     - received user_agent={user_agent}")
    return "Hello World"  # TODO: Implement


@web_app.post("/api/v1/classify")
async def handle_classification(input_: ModelInput, user_agent: Optional[str] = Header(None)):
    """
    Classify a body of text as spam or ham.

    eg. 
    
    ```bash
    curl -X POST https://modal-labs--example-spam-detect-llm-fastapi-app-thun-ed2e0f-dev.modal.run/api/v1/classify \ 
    -H 'Content-Type: application/json' \
    -d '{"text": "hello world"}'
    ```
    """
    # TODO(Jonathon): Cache this information in a modal.Dict, with TTL.
    prediction = classifier(input_.text)
    return ModelOutput(
        spam=prediction.spam,
        score=prediction.score,
    )


@stub.asgi(
    shared_volumes={config.VOLUME_DIR: volume},
)
def fastapi_app():
    return web_app


if __name__ == "__main__":
    stub.serve()
