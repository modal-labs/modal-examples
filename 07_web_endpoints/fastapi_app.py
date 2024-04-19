# ---
# lambda-test: false
# ---

from typing import Optional

from fastapi import FastAPI, Header
from modal import App, Image, asgi_app, web_endpoint
from pydantic import BaseModel

web_app = FastAPI()
app = App("example-fastapi-app")
image = Image.debian_slim()


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
@asgi_app()
def fastapi_app():
    return web_app


@app.function()
@web_endpoint(method="POST")
def f(item: Item):
    return "Hello " + item.name


if __name__ == "__main__":
    app.deploy("webapp")
