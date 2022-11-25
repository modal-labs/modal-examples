"""
Defines a serverless web API to expose trained models 
"""
from typing import Optional

import modal
from fastapi import FastAPI, Header
from pydantic import BaseModel

from . import config
from .app import stub, volume

web_app = FastAPI()


class Input(BaseModel):
    text: str


@web_app.get("/api/v1/models")
async def handle_list_models(user_agent: Optional[str] = Header(None)):
    """
    Show details of actively serving models.
    """
    print(f"GET /     - received user_agent={user_agent}")
    return "Hello World"


@web_app.post("/api/v1/classify")
async def handle_classification(input: Input, user_agent: Optional[str] = Header(None)):
    """
    Classify a body of text as spam or ham.
    """
    print(f"POST /foo - received user_agent={user_agent}, {input.text=}")
    # TODO(Jonathon):
    # - Load model serving config from a file
    # - Cache this information in a modal.Dict, with TTL.
    return input


@stub.asgi(
    shared_volumes={config.VOLUME_DIR: volume},
)
def fastapi_app():
    return web_app


if __name__ == "__main__":
    stub.serve()
