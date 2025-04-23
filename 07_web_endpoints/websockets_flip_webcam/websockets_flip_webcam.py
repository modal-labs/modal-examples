import modal
from pathlib import Path
from typing import Any
from dataclasses import dataclass

this_directory = Path(__file__).parent.resolve()

image = (
    modal.Image.debian_slim()
    .apt_install("python3-opencv", "ffmpeg")
    .run_commands("pip install --upgrade pip")
    .pip_install("fastapi", "websockets", "gradio", "opencv-python")
    .add_local_dir(this_directory, remote_path="/assets")
)
app = modal.App(
    "websockets-flip-webcam",
    image=image,
)

@dataclass
class Frame:
    client_id: str
    image: Any # Using Any since np.ndarray requires numpy import


@app.cls(
    region="ap-south",
)
@modal.concurrent(max_inputs=10)
class WebsocketsFlipWebcam:

    @modal.enter()
    async def init(self):

        import asyncio

        self.last_frame_time = {}
        self.latest_frame = {}
        self.delay_msec = {}

        self.frame_queue = asyncio.Queue()
        self.frame_processor_task = asyncio.create_task(self.handle_queue())
        
        self.websockets = {}
        

    async def flip_vertically(self, image, delay_msec = 0., round_trip_time = 0.):

        import asyncio
        import numpy as np
        import cv2

        if delay_msec > 0:
            await asyncio.sleep(delay_msec / 1000)

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

        import numpy as np
        import cv2
        import time

        print("Frame processor task started")
        
        while True:

            try:

                print("Frame processor task processing frame queue")
                
                frame = await self.frame_queue.get()
                client_id = frame.client_id

                if client_id not in self.websockets:
                    print(f"Client {client_id} not in websockets, skipping frame")
                    continue

                frame = self.latest_frame.pop(client_id, None)
                if frame:
                    image = frame.image

                
                if image is not None:

                    print(f"Frame processor task processing frame for client {client_id}")

                    now = time.time()
                    if self.last_frame_time[client_id] is None:
                        round_trip_time = np.nan
                    else:
                        round_trip_time = now - self.last_frame_time[client_id]
                    self.last_frame_time[client_id] = now

                    delay_msec = self.delay_msec[client_id]

                    new_image = await self.flip_vertically(image, delay_msec = delay_msec, round_trip_time = round_trip_time)

                    # Convert back to bytes
                    success, buffer = cv2.imencode('.jpg', new_image, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    if not success:
                        print("Failed to encode image")
                        continue

                    print(f"Frame processor task sending frame to client {client_id}")
                    await self.websockets[client_id].send_bytes(buffer.tobytes())

            except Exception as e:
                print(f"Frame processor task exception: {e.with_traceback()}")


        

                
    @modal.exit()
    async def exit(self):
        if not self.websockets:
            print("No websockets, exiting frame processor task")
            self.frame_processor_task.cancel()


        

    @modal.asgi_app()
    def endpoint(self):

        import numpy as np
        import cv2
        import json

        from fastapi import WebSocketDisconnect

        from fastapi import FastAPI, WebSocket, Request
        from fastapi.websockets import WebSocketState
        from fastapi.responses import HTMLResponse

        web_app = FastAPI()

        @web_app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str) -> None:
            
            print(f"WebSocket connection requested for client {client_id}")
            await websocket.accept()

            self.websockets[client_id] = websocket
            self.delay_msec[client_id] = 0.
            self.last_frame_time[client_id] = None

            print("WebSocket connection accepted")

            while True: 

                try:

                    data = await websocket.receive_bytes()   
                    if data and isinstance(data, bytes):
                        print(f"Received {len(data)} bytes from client {client_id}")
                        # Convert bytes to numpy array
                        nparr = np.frombuffer(data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        self.latest_frame[client_id] = Frame(client_id=client_id, image=img)
                        self.frame_queue.put_nowait(Frame(client_id=client_id, image=img))

                        continue
                    
                except Exception as e:

                    print(f"Websocket handling exception for client {client_id}: {e}")
                    if isinstance(e, WebSocketDisconnect):
                        print(f"Websocket {client_id} disconnected")
                        await websocket.close()
                        self.websockets.pop(client_id)
                        return
                    
                try:

                    data = await websocket.receive_text()
                    if data and isinstance(data, str):
                        try:
                            json_data = json.loads(data)

                            if json_data["type"] == "delay":
                                print("Setting delay to", json_data["value"])
                                self.delay_msec[client_id] = json_data["value"]
                                continue

                        except Exception as e:
                            print("Failed to parse websocket string message as json", e)

                except Exception as e:

                    print(f"Websocket handling exception for client {client_id}: {e}")
                    if isinstance(e, WebSocketDisconnect):
                        print(f"Websocket {client_id} disconnected")
                        await websocket.close()
                        self.websockets.pop(client_id)
                        return

        @web_app.get("/")
        async def get(request: Request):

            html = open("/assets/index.html").read()
            base_url = str(request.url).replace("http", "ws")
            html = html.replace("url-placeholder", f"{base_url}ws")
            return HTMLResponse(html)

        return web_app

