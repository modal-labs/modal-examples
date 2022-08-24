import modal

stub = modal.Stub("web-flask", image=modal.DebianSlim().pip_install(["flask", "asgiref"]))


@stub.asgi
def flask_app():
    from asgiref.wsgi import WsgiToAsgi
    from flask import Flask, request

    web_app = Flask(__name__)

    @web_app.get("/")
    def home():
        return "Hello Flask World!"

    @web_app.post("/foo")
    def foo():
        return request.json

    return WsgiToAsgi(web_app)


if __name__ == "__main__":
    stub.run_forever()
