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
    region="ap-south",
)
class WebsocketsFlipWebcam:

    @modal.enter()
    def init(self):

        import asyncio

        self.last_frame_time = None
        self.delay_msec = 0

        self.frame_queue = asyncio.Queue()
        self.frame_processor_task = None
        self.latest_frame = None
        
        self.websocket = None

    async def flip_vertically(self, image):

        import asyncio
        import time
        import numpy as np
        import cv2

        now = time.time()
        if self.last_frame_time is None:
            round_trip_time = np.nan
        else:
            round_trip_time = now - self.last_frame_time
        self.last_frame_time = now

        if self.delay_msec > 0:
            await asyncio.sleep(self.delay_msec / 1000)

        img = image.astype(np.uint8)
                
        if img is None:
            print("Failed to decode image")
            return None
            
        print(f"Image shape: {img.shape}")
        
        # Flip vertically
        flipped = cv2.flip(img, 0)

        # add round trip time to image
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
    
    async def handle_queue(self):

        import cv2
        import asyncio
        
        while True:

            if self.websocket:
                
                image = await self.frame_queue.get()
                image = self.latest_frame
                
                if image is not None:
                    new_image = await self.flip_vertically(image)

                    # Convert back to bytes
                    success, buffer = cv2.imencode('.jpg', new_image, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    if not success:
                        print("Failed to encode image")
                        continue

                    await self.websocket.send_bytes(buffer.tobytes())

            else:
                await asyncio.sleep(0.250)

    @modal.asgi_app()
    def endpoint(self):

        import asyncio
        import numpy as np
        import cv2
        import json
        from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
        from fastapi.responses import HTMLResponse

        web_app = FastAPI()

        

        @web_app.websocket("/ws")
        async def websocket_handler(websocket: WebSocket) -> None:

            await websocket.accept()
            self.websocket = websocket

            self.frame_processor_task = asyncio.create_task(self.handle_queue())
            
            print("WebSocket connection accepted")
            
            try:
                while True:
                    try:
                        data = await websocket.receive_bytes()                    
                        # Convert bytes to numpy array
                        nparr = np.frombuffer(data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        self.latest_frame = img
                        self.frame_queue.put_nowait(img)

                        continue

                        
                    except Exception as e:
                        if isinstance(e, WebSocketDisconnect):
                            raise WebSocketDisconnect
                        
                    try:
                        data = await websocket.receive_text()
                        if data and isinstance(data, str):
                            try:
                                json_data = json.loads(data)

                                if json_data["type"] == "delay":
                                    print("Setting delay to", json_data["value"])
                                    self.delay_msec = json_data["value"]
                                    continue
                            except Exception as e:
                                print("Failed to parse websocket message as json", e)
                    except Exception as e:
                        
                        if isinstance(e, WebSocketDisconnect):
                            raise WebSocketDisconnect
                        
            except WebSocketDisconnect as e:
                print("handle disconnect")
                await self.websocket.close()
                self.frame_processor_task.cancel()

        @web_app.get("/")
        async def get(request: Request):
            html = open("/assets/index.html").read()
            base_url = str(request.url).replace("http", "ws")
            html = html.replace("url-placeholder", f"{base_url}ws")
            return HTMLResponse(html)

        return web_app

