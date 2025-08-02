# ---
# cmd: ["modal", "serve", "07_web_endpoints/flask_app.py"]
# ---

# # Deploy Flask app with Modal

# This example shows how you can deploy a [Flask](https://flask.palletsprojects.com/en/3.0.x/) app with Modal.
# You can serve any app written in a WSGI-compatible web framework (like Flask) on Modal with this pattern. You can serve an app written in an ASGI-compatible framework, like FastAPI, with [`asgi_app`](https://modal.com/docs/guide/webhooks#asgi).

import modal

app = modal.App(
    "example-flask-app",
    image=modal.Image.debian_slim().pip_install("flask"),
)


@app.function()
@modal.wsgi_app()
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
