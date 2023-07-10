"""
Defines a serverless web API to expose trained models 
"""
from typing import Optional

import modal
from fastapi import FastAPI, Header
from modal.cls import ClsMixin
from pydantic import BaseModel

from . import config
from . import models
from .app import stub, volume

web_app = FastAPI()


class ModelInput(BaseModel):
    text: str


class ModelMetdata(BaseModel):
    model_name: str
    model_id: str


class ModelOutput(BaseModel):
    spam: bool
    score: float
    metadata: ModelMetdata


# TODO(Jonathon): This will acquire a GPU even when `model_id` doesn't
# require it, which is inefficient. Find an elegant way to make the GPU optional.
@stub.cls(gpu="A10G", volumes={config.VOLUME_DIR: volume})
class Model(ClsMixin):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        classifier, metadata = models.load_model(model_id=self.model_id)
        self.classifier = classifier
        self.metadata = metadata

    @modal.method()
    def generate(self, text: str) -> ModelOutput:
        prediction = self.classifier(text)
        return ModelOutput(
            spam=prediction.spam,
            score=prediction.score,
            metadata=ModelMetdata(
                model_name=self.metadata.impl_name,
                model_id=self.model_id,
            ),
        )


@web_app.get("/api/v1/models")
async def handle_list_models():
    """
    Show details of actively serving models.
    """
    _, metadata = models.load_model(config.SERVING_MODEL_ID)
    return {config.SERVING_MODEL_ID: metadata.serialize()}


@web_app.post("/api/v1/classify")
async def handle_classification(
    input_: ModelInput, model_id: Optional[str] = Header(None)
):
    """
    Classify a body of text as spam or ham.

    eg. 
    
    ```bash
    curl -X POST https://modal-labs--example-spam-detect-llm-web.modal.run/api/v1/classify \ 
    -H 'Content-Type: application/json' \
    -H 'Model-Id: sha256.12E5065BE4C3F7D2F79B7A0FD203380869F6E308DCBB4B8C9579FFAE6F32B837' \
    -d '{"text": "hello world"}'
    ```
    """
    model_id = model_id or config.SERVING_MODEL_ID
    print(model_id)
    model = Model.remote(model_id)
    return model.generate(input_.text)


@stub.function()
@modal.asgi_app()
def web():
    return web_app


if __name__ == "__main__":
    stub.serve()
