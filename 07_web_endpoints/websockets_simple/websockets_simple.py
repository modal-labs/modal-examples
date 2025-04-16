import modal
from pathlib import Path

static_path = Path(__file__).parent.resolve()

app = modal.App("websockets-simple-text-echo")
app.image = modal.Image.debian_slim().run_commands("pip install --upgrade pip").pip_install("fastapi", "websockets", "gradio").add_local_dir(static_path, remote_path="/assets")

@app.function()
@modal.asgi_app()
def endpoint():
    from fastapi import FastAPI, WebSocket, Request
    from fastapi.responses import HTMLResponse

    web_app = FastAPI()

    @web_app.websocket("/ws")
    async def websocket_handler(websocket: WebSocket) -> None:
        await websocket.accept()
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")

    @web_app.get("/")
    async def get(request: Request):
        html = open("/assets/index.html").read()
        # Get the current URL and replace the protocol with wss
        base_url = str(request.url).replace("http", "ws")
        # Replace the hardcoded URL with the dynamic one
        html = html.replace("url-placeholder", f"{base_url}ws")
        return HTMLResponse(html)

    return web_app

