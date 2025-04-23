import modal
from pathlib import Path

this_folder = Path(__file__).parent.resolve()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("python3-opencv", "ffmpeg")
    .env(
        {
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.12/site-packages/tensorrt_libs",
        }
    )
    .run_commands("pip install --upgrade pip")
    .pip_install(
        "fastapi", 
        "websockets", 
        "gradio", 
        "opencv-python",
        "tensorrt",
        "torch",
        "onnxruntime-gpu"
    )
    .add_local_dir(this_folder, remote_path="/assets")
)

app = modal.App(
    "websockets-yolo-demo",
    image=image,
)

@app.cls(
    gpu="A100",
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
class WebsocketsYOLODemo:


    @modal.enter()
    def load_model(self):

        import asyncio
        import time
        import numpy as np

        from huggingface_hub import hf_hub_download
        import cv2
        import onnxruntime

        self.last_frame_time = None
        self.conf_threshold = 0.15
        self.delay_msec = 0
        self.frame_queue = asyncio.Queue()
        self.latest_frame = None
        self.websocket = None

        onnxruntime.preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None)


        class YOLOv10:
            def __init__(self, path):
                # Initialize model
                model_file = hf_hub_download(
                    repo_id="onnx-community/yolov10n", filename="onnx/model.onnx"
                )
                self.initialize_model(model_file)

            # def __call__(self, image):
            #     return self.detect_objects(image)

            def initialize_model(self, path):
                self.session = onnxruntime.InferenceSession(
                    path, providers=onnxruntime.get_available_providers()
                )
                # Get model info
                self.get_input_details()
                self.get_output_details()
                rng = np.random.default_rng(3)
                self.colors = rng.uniform(0, 255, size=(len(class_names), 3))

            def detect_objects(self, image, conf_threshold=0.3):
                input_tensor = self.prepare_input(image)

                # Perform inference on the image
                new_image = self.inference(image, input_tensor, conf_threshold)

                return new_image

            def prepare_input(self, image):
                self.img_height, self.img_width = image.shape[:2]

                input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Resize input image
                input_img = cv2.resize(input_img, (self.input_width, self.input_height))

                # Scale input pixel values to 0 to 1
                input_img = input_img / 255.0
                input_img = input_img.transpose(2, 0, 1)
                input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

                return input_tensor

            def inference(self, image, input_tensor, conf_threshold=0.3):
                start = time.perf_counter()
                outputs = self.session.run(
                    self.output_names, {self.input_names[0]: input_tensor}
                )

                print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
                (
                    boxes,
                    scores,
                    class_ids,
                ) = self.process_output(outputs, conf_threshold)
                return self.draw_detections(image, boxes, scores, class_ids)

            def process_output(self, output, conf_threshold=0.3):
                predictions = np.squeeze(output[0])

                # Filter out object confidence scores below threshold
                scores = predictions[:, 4]
                predictions = predictions[scores > conf_threshold, :]
                scores = scores[scores > conf_threshold]

                if len(scores) == 0:
                    return [], [], []

                # Get the class with the highest confidence
                class_ids = predictions[:, 5].astype(int)

                # Get bounding boxes for each object
                boxes = self.extract_boxes(predictions)

                return boxes, scores, class_ids

            def extract_boxes(self, predictions):
                # Extract boxes from predictions
                boxes = predictions[:, :4]

                # Scale boxes to original image dimensions
                boxes = self.rescale_boxes(boxes)

                # Convert boxes to xyxy format
                # boxes = xywh2xyxy(boxes)

                return boxes

            def rescale_boxes(self, boxes):
                # Rescale boxes to original image dimensions
                input_shape = np.array(
                    [self.input_width, self.input_height, self.input_width, self.input_height]
                )
                boxes = np.divide(boxes, input_shape, dtype=np.float32)
                boxes *= np.array(
                    [self.img_width, self.img_height, self.img_width, self.img_height]
                )
                return boxes

            def draw_detections(
                self, image, boxes, scores, class_ids, draw_scores=True, mask_alpha=0.4
            ):
                det_img = image.copy()

                img_height, img_width = image.shape[:2]
                font_size = min([img_height, img_width]) * 0.0006
                text_thickness = int(min([img_height, img_width]) * 0.001)

                # det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

                # Draw bounding boxes and labels of detections
                for class_id, box, score in zip(class_ids, boxes, scores):
                    color = self.colors[class_id]

                    self.draw_box(det_img, box, color)  # type: ignore

                    label = class_names[class_id]
                    caption = f"{label} {int(score * 100)}%"
                    self.draw_text(det_img, caption, box, color, font_size, text_thickness)  # type: ignore

                return det_img

            def get_input_details(self):
                model_inputs = self.session.get_inputs()
                self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

                self.input_shape = model_inputs[0].shape
                self.input_height = self.input_shape[2]
                self.input_width = self.input_shape[3]

            def get_output_details(self):
                model_outputs = self.session.get_outputs()
                self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

            def draw_box(
                self,
                image: np.ndarray,
                box: np.ndarray,
                color: tuple[int, int, int] = (0, 0, 255),
                thickness: int = 2,
            ) -> np.ndarray:
                x1, y1, x2, y2 = box.astype(int)
                return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


            def draw_text(
                self,
                image: np.ndarray,
                text: str,
                box: np.ndarray,
                color: tuple[int, int, int] = (0, 0, 255),
                font_size: float = 0.001,
                text_thickness: int = 2,
            ) -> np.ndarray:
                x1, y1, x2, y2 = box.astype(int)
                (tw, th), _ = cv2.getTextSize(
                    text=text,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_size,
                    thickness=text_thickness,
                )
                th = int(th * 1.2)

                cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)

                return cv2.putText(
                    image,
                    text,
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (255, 255, 255),
                    text_thickness,
                    cv2.LINE_AA,
                )
        
    
        self.model = YOLOv10("yolov10m.pt")

    async def handle_queue(self):

        import cv2

        if self.websocket:
            while True:
                
                image = await self.frame_queue.get()
                image = self.latest_frame
                
                new_image = await self.detection(image)

                # Convert back to bytes
                success, buffer = cv2.imencode('.jpg', new_image, [cv2.IMWRITE_JPEG_QUALITY, 50])
                if not success:
                    print("Failed to encode image")
                    continue

                await self.websocket.send_bytes(buffer.tobytes())

    async def detection(self, image):

        import asyncio
        import time
        import cv2

        print(self.conf_threshold, self.delay_msec)

        now = time.time()
        if self.last_frame_time is None:
            round_trip_time = 0.
        else:
            round_trip_time = now - self.last_frame_time
        self.last_frame_time = now

        if self.delay_msec > 0:
            time.sleep(self.delay_msec / 1000)

        print(f"Image shape: {image.shape}")
        image = cv2.resize(image, (self.model.input_width, self.model.input_height))
        print("conf_threshold", self.conf_threshold)
        new_image = self.model.detect_objects(image, self.conf_threshold)
        new_image = cv2.resize(new_image, (500, 500))
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
        text1_x = new_image.shape[1] - text1_width - margin
        text1_y = new_image.shape[0] - text1_height - margin - text2_height
        text2_x = new_image.shape[1] - text2_width - margin  
        text2_y = new_image.shape[0] - margin

        # Draw both lines of text
        cv2.putText(new_image, text1, (text1_x, text1_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0, 128), thickness)
        cv2.putText(new_image, text2, (text2_x, text2_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0, 128), thickness)

        return new_image

    @modal.asgi_app()
    def endpoint(self):

        import json

        import numpy as np
        import cv2

        import asyncio

        from fastapi import FastAPI, WebSocket, Request
        from fastapi.responses import HTMLResponse
        from fastapi import WebSocketDisconnect

        web_app = FastAPI()

        

        @web_app.websocket("/ws")
        async def websocket_handler(websocket: WebSocket) -> None:

            
            await websocket.accept()
            self.websocket = websocket

            asyncio.create_task(self.handle_queue())

            print("WebSocket connection accepted")
            try:
                while True:
                    
                    try:
                        data = await self.websocket.receive_bytes()

                        if data and isinstance(data, bytes):
                            image = None
                            try:
                                # Convert bytes to numpy array
                                nparr = np.frombuffer(data, np.uint8)
                                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                            except:
                                print("Failed to decode websocket message")
                                continue 

                            self.latest_frame = image
                            self.frame_queue.put_nowait(image)

                            continue

                    except Exception as e:
                        
                        if isinstance(e, WebSocketDisconnect):
                            raise WebSocketDisconnect

                    try:

                        data = await self.websocket.receive_text()
                        if data and isinstance(data, str):
                            try:
                                
                                # try parsing as json
                                json_data = json.loads(data)
                                
                                print("Received json data:", json_data)
                                
                                if json_data["type"] == "confidenceThresh":
                                    print("Setting confidence threshold to", json_data["value"])
                                    self.conf_threshold = json_data["value"]
                                    continue
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

                


        @web_app.get("/")
        async def get(request: Request):
            html = open("/assets/index.html").read()
            base_url = str(request.url).replace("http", "ws")
            html = html.replace("url-placeholder", f"{base_url}ws")
            return HTMLResponse(html)

        return web_app

