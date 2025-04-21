import modal
from pathlib import Path

this_directory = Path(__file__).parent.resolve()

app = modal.App("websockets-flip-webcam")
app.image = (
    modal.Image.debian_slim()
    .apt_install("python3-opencv", "ffmpeg")
    .run_commands("pip install --upgrade pip")
    .pip_install("fastapi", "websockets", "gradio", "opencv-python")
    .add_local_dir(this_directory, remote_path="/assets")
)

@app.cls(
    # region="ap-south"
)
@modal.concurrent(max_inputs=100)
class WebsocketsFlipWebcam:

    @modal.enter()
    def init(self):
        self.last_frame_time = None

    @modal.asgi_app()
    def endpoint(self):

        import time
        import numpy as np
        import cv2

        from fastapi import FastAPI, WebSocket, Request
        from fastapi.responses import HTMLResponse

        web_app = FastAPI()

        def flip_vertically(image):
            now = time.time()
            if self.last_frame_time is None:
                round_trip_time = np.nan
            else:
                round_trip_time = now - self.last_frame_time
            self.last_frame_time = now

            img = image.astype(np.uint8)
                    
            if img is None:
                print("Failed to decode image")
                return None
                
            print(f"Image shape: {img.shape}")
            
            # Flip vertically
            flipped = cv2.flip(img, 0)

            # add round trip time to image
            # Get text size to position it in lower right
            # Split text into two lines and render separately
            text1 = "Round trip time:"
            text2 = f"{round_trip_time*1000:>6.1f} msec"
            font_scale = 0.8
            thickness = 2
            
            # Get text sizes
            (text1_width, text1_height), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            (text2_width, text2_height), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Position text in bottom right, with text2 below text1
            margin = 10
            text1_x = flipped.shape[1] - text1_width - margin
            text1_y = flipped.shape[0] - text1_height - margin - text2_height
            text2_x = flipped.shape[1] - text2_width - margin  
            text2_y = flipped.shape[0] - margin

            # Draw both lines of text
            cv2.putText(flipped, text1, (text1_x, text1_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0, 128), thickness)
            cv2.putText(flipped, text2, (text2_x, text2_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0, 128), thickness)

            return flipped

        @web_app.websocket("/ws")
        async def websocket_handler(websocket: WebSocket) -> None:

            await websocket.accept()
            
            print("WebSocket connection accepted")
            
            while True:
                try:
                    data = await websocket.receive_bytes()                    
                    # Convert bytes to numpy array
                    nparr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    print(type(img))
                    flipped = flip_vertically(img)

                    # Convert back to bytes
                    success, buffer = cv2.imencode('.jpg', flipped, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    if not success:
                        print("Failed to encode image")
                        continue
                        
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

