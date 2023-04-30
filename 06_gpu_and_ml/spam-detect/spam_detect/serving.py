"""
Defines a serverless web API to expose trained models 
"""
from typing import Optional

import modal
from modal import method
from fastapi import FastAPI, Header
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


@stub.cls(shared_volumes={config.VOLUME_DIR: volume})
class Model:
    def __enter__(self):
        classifier, metadata = models.load_model(
            model_id=config.SERVING_MODEL_ID
        )
        self.classifier = classifer
        self.metadata = metadata

    @web_app.get("/api/v1/models")
    def handle_list_models(self, user_agent: Optional[str] = Header(None)):
        """
        Show details of actively serving models.
        """
        return {config.SERVING_MODEL_ID: self.metadata.serialize()}

    @web_app.post("/api/v1/classify")
    def handle_classification(
        self, input_: ModelInput, user_agent: Optional[str] = Header(None)
    ):
        """
        Classify a body of text as spam or ham.

        **Example:**
        
        ```bash
        curl -X POST https://modal-labs--example-spam-detect-llm-web.modal.run/api/v1/classify \ 
        -H 'Content-Type: application/json' \
        -d '{"text": "hello world"}'
        ```
        """
        prediction = self.classifier(input_.text)
        return ModelOutput(
            spam=prediction.spam,
            score=prediction.score,
            metadata=ModelMetdata(
                model_name=metadata.impl_name,
                model_id=config.SERVING_MODEL_ID,
            ),
        )

    @method()
    @modal.asgi_app()
    def web(self):
        return web_app


if __name__ == "__main__":
    stub.serve()
