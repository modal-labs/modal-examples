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
    classifier, metadata = models.load_model(model_id=config.SERVING_MODEL_ID)
else:
    classifier, metadata = None, None


class ModelInput(BaseModel):
    text: str


class ModelMetdata(BaseModel):
    model_name: str
    model_id: str


class ModelOutput(BaseModel):
    spam: bool
    score: float
    metadata: ModelMetdata


@web_app.get("/api/v1/models")
async def handle_list_models(user_agent: Optional[str] = Header(None)):
    """
    Show details of actively serving models.
    """
    return {config.SERVING_MODEL_ID: metadata.serialize()}


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
    prediction = classifier(input_.text)
    return ModelOutput(
        spam=prediction.spam,
        score=prediction.score,
        metadata=ModelMetdata(
            model_name=metadata.impl_name,
            model_id=config.SERVING_MODEL_ID,
        ),
    )


@stub.asgi(
    shared_volumes={config.VOLUME_DIR: volume},
)
def fastapi_app():
    return web_app


if __name__ == "__main__":
    stub.serve()
