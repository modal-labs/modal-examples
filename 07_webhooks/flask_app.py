# ---
# lambda-test: false
# ---

import modal

stub = modal.Stub(
    "example-web-flask",
    image=modal.Image.debian_slim().pip_install("flask"),
)


@stub.wsgi
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
