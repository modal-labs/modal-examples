import modal
from app.serve import app

stub = modal.Stub("ace-step-web")
web_app = modal.asgi_app(app)

stub.asgi_app(web_app)