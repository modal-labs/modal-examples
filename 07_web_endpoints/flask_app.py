# ---
# lambda-test: false
# ---

from modal import App, Image, wsgi_app

app = App(
    "example-web-flask",
    image=Image.debian_slim().pip_install("flask"),
)


@app.function()
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
