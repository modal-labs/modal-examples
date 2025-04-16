import modal
from pathlib import Path

static_path = Path(__file__).parent.resolve()

app = modal.App("websockets-video-echo")
app.image = (
    modal.Image.debian_slim()
    .apt_install("python3-opencv", "ffmpeg")
    .run_commands("pip install --upgrade pip")
    .pip_install("fastapi", "websockets", "gradio", "opencv-python")
    .add_local_dir(static_path, remote_path="/assets")
)

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
        import numpy as np
        import cv2

        web_app = FastAPI()

        @web_app.websocket("/ws")
        async def websocket_handler(websocket: WebSocket) -> None:
            await websocket.accept()
            print("WebSocket connection accepted")
            
            while True:
                try:
                    data = await websocket.receive_bytes()
                    print(f"Received {len(data)} bytes")
                    
                    # Convert bytes to numpy array
                    nparr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        print("Failed to decode image")
                        continue
                        
                    print(f"Image shape: {img.shape}")
                    
                    # Flip vertically
                    flipped = cv2.flip(img, 0)
                    
                    # Convert back to bytes
                    success, buffer = cv2.imencode('.jpg', flipped, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    if not success:
                        print("Failed to encode image")
                        continue
                        
                    print(f"Sending {len(buffer)} bytes")
                    await websocket.send_bytes(buffer.tobytes())
                    
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    break

        @web_app.get("/")
        async def get(request: Request):
            html = open("/assets/index.html").read()
            base_url = str(request.url).replace("http", "ws")
            html = html.replace("url-placeholder", f"{base_url}ws")
            return HTMLResponse(html)

        return web_app

