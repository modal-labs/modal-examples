# ---
# lambda-test: false
# ---

# # Deploy FastAPI app with Modal

# This example shows how you can deploy a [FastAPI](https://fastapi.tiangolo.com/) app with Modal.
# You can serve any app written in an ASGI or WSGI-compatible web framework (like FastAPI) on Modal.

from typing import Optional

import modal
from fastapi import FastAPI, Header
from pydantic import BaseModel

web_app = FastAPI()
app = modal.App("example-fastapi-app")
image = modal.Image.debian_slim()


class Item(BaseModel):
    name: str


@web_app.get("/")
async def handle_root(user_agent: Optional[str] = Header(None)):
    print(f"GET /     - received user_agent={user_agent}")
    return "Hello World"


@web_app.post("/foo")
async def handle_foo(item: Item, user_agent: Optional[str] = Header(None)):
    print(
        f"POST /foo - received user_agent={user_agent}, item.name={item.name}"
    )
    return item


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app


@app.function()
@modal.web_endpoint(method="POST")
def f(item: Item):
    return "Hello " + item.name


if __name__ == "__main__":
    app.deploy("webapp")
