# ---
# lambda-test: false
# ---

from modal import Image, Stub, wsgi_app

stub = Stub(
    "example-web-flask",
    image=Image.debian_slim().pip_install("flask"),
)


@stub.function()
@wsgi_app()
def flask_app():
    from flask import Flask, request

    web_app = Flask(__name__)

    @web_app.get("/")
    def home():
        return "Hello Flask World!"

    @web_app.post("/foo")
    def foo():
        return request.json

    return web_app
