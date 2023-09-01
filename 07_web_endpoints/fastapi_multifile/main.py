import modal
from fastapi import FastAPI

from .routers import items, users

app = FastAPI()

stub = modal.Stub("fastapi-multifile-example")

image = modal.Image.debian_slim().pip_install("pandas")


app.include_router(users.router)
app.include_router(items.router)


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}


@stub.function(image=image)
@modal.asgi_app(custom_domains=["potatoes.ai"])
def fastapi_app():
    return app
