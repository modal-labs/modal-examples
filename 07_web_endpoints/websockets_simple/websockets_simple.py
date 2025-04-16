import modal
from pathlib import Path

static_path = Path(__file__).parent.resolve()

app = modal.App("websockets-video-echo")
app.image = modal.Image.debian_slim().run_commands("pip install --upgrade pip").pip_install("fastapi", "websockets", "gradio").add_local_dir(static_path, remote_path="/assets")

@app.cls()
@modal.concurrent(max_inputs=100)
class VideoEcho:
    @modal.enter()
    def init(self):
        pass

    @modal.asgi_app()
    def endpoint(self):
        from fastapi import FastAPI, WebSocket, Request
        from fastapi.responses import HTMLResponse

        web_app = FastAPI()

        @web_app.websocket("/ws")
        async def websocket_handler(websocket: WebSocket) -> None:
            await websocket.accept()
            while True:
                data = await websocket.receive_text()
                # Simply echo back the image data
                await websocket.send_text(data)

        @web_app.get("/")
        async def get(request: Request):
            html = open("/assets/index.html").read()
            base_url = str(request.url).replace("http", "ws")
            html = html.replace("url-placeholder", f"{base_url}ws")
            return HTMLResponse(html)

        return web_app

