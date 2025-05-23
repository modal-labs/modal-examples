# ---
# cmd: ["modal", "serve", "07_web_endpoints/fastapi_app.py"]
# ---

# # Deploy FastAPI app with Modal

# This example shows how you can deploy a [FastAPI](https://fastapi.tiangolo.com/) app with Modal.
# You can serve any app written in an ASGI-compatible web framework (like FastAPI) using this pattern or you can server WSGI-compatible frameworks like Flask with [`wsgi_app`](https://modal.com/docs/guide/webhooks#wsgi).

from typing import Optional

import modal
from fastapi import FastAPI, Header
from pydantic import BaseModel

image = modal.Image.debian_slim().pip_install("fastapi[standard]", "pydantic")
app = modal.App("example-fastapi-app", image=image)
web_app = FastAPI()


class Item(BaseModel):
    name: str


@web_app.get("/")
async def handle_root(user_agent: Optional[str] = Header(None)):
    print(f"GET /     - received user_agent={user_agent}")
    return "Hello World"


@web_app.post("/foo")
async def handle_foo(item: Item, user_agent: Optional[str] = Header(None)):
    print(f"POST /foo - received user_agent={user_agent}, item.name={item.name}")
    return item


@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app


@app.function()
@modal.fastapi_endpoint(method="POST")
def f(item: Item):
    return "Hello " + item.name


if __name__ == "__main__":
    app.deploy("webapp")
