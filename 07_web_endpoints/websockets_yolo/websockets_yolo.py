from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal

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
        "onnxruntime-gpu",
    )
    .add_local_dir(this_folder, remote_path="/assets")
)

app = modal.App(
    "websockets-yolo-demo",
    image=image,
)


@dataclass
class Frame:
    client_id: str
    image: Any  # Using Any since np.ndarray requires numpy import


@app.cls(
    gpu="A100",
)
@modal.concurrent(max_inputs=10)
class WebsocketsYOLODemo:
    @modal.enter()
    def init(self):
        import asyncio
        import time

        import cv2
        import numpy as np
        import onnxruntime
        from huggingface_hub import hf_hub_download

        self.last_frame_time = {}
        self.latest_frame = {}
        self.delay_msec = {}
        self.conf_threshold = {}

        self.frame_queue = asyncio.Queue()
        self.frame_processor_task = None
        self.websockets = {}

        onnxruntime.preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None)

        class YOLOv10:
            def __init__(self, path):
                # Initialize model
                model_file = hf_hub_download(
                    repo_id="onnx-community/yolov10n",
                    filename="onnx/model.onnx",
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

                # get class names
                with open("/assets/yolo_classes.txt", "r") as f:
                    self.class_names = f.read().splitlines()
                rng = np.random.default_rng(3)
                self.colors = rng.uniform(0, 255, size=(len(self.class_names), 3))

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
                    [
                        self.input_width,
                        self.input_height,
                        self.input_width,
                        self.input_height,
                    ]
                )
                boxes = np.divide(boxes, input_shape, dtype=np.float32)
                boxes *= np.array(
                    [
                        self.img_width,
                        self.img_height,
                        self.img_width,
                        self.img_height,
                    ]
                )
                return boxes

            def draw_detections(
                self,
                image,
                boxes,
                scores,
                class_ids,
                draw_scores=True,
                mask_alpha=0.4,
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

                    label = self.class_names[class_id]
                    caption = f"{label} {int(score * 100)}%"
                    self.draw_text(
                        det_img, caption, box, color, font_size, text_thickness
                    )  # type: ignore

                return det_img

            def get_input_details(self):
                model_inputs = self.session.get_inputs()
                self.input_names = [
                    model_inputs[i].name for i in range(len(model_inputs))
                ]

                self.input_shape = model_inputs[0].shape
                self.input_height = self.input_shape[2]
                self.input_width = self.input_shape[3]

            def get_output_details(self):
                model_outputs = self.session.get_outputs()
                self.output_names = [
                    model_outputs[i].name for i in range(len(model_outputs))
                ]

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

    @modal.exit()
    async def exit(self):
        if not self.websockets:
            print("No websockets, exiting frame processor task")
            self.frame_processor_task.cancel()

    async def handle_queue(self):
        import time

        import cv2
        import numpy as np

        while True:
            try:
                frame = await self.frame_queue.get()
                client_id = frame.client_id

                if client_id not in self.websockets:
                    print(f"Client {client_id} not in websockets, skipping frame")
                    continue

                frame = self.latest_frame.pop(client_id, None)
                if frame:
                    image = frame.image

                if image is not None:
                    print(
                        f"Frame processor task processing frame for client {client_id}"
                    )

                    now = time.time()
                    if self.last_frame_time[client_id] is None:
                        round_trip_time = np.nan
                    else:
                        round_trip_time = now - self.last_frame_time[client_id]
                    self.last_frame_time[client_id] = now

                    delay_msec = self.delay_msec[client_id]
                    conf_threshold = self.conf_threshold[client_id]

                image_w_detections = await self.detection(
                    image, conf_threshold, round_trip_time, delay_msec
                )

                # Convert back to bytes
                success, buffer = cv2.imencode(
                    ".jpg", image_w_detections, [cv2.IMWRITE_JPEG_QUALITY, 50]
                )
                if not success:
                    print("Failed to encode image")
                    continue

                await self.websockets[client_id].send_bytes(buffer.tobytes())

            except Exception as e:
                print(f"Error in handle_queue: {e}")

    async def detection(self, image, conf_threshold, round_trip_time, delay_msec):
        import asyncio

        import cv2

        if delay_msec > 0:
            await asyncio.sleep(delay_msec / 1000)

        print(f"Image shape: {image.shape}")
        image = cv2.resize(image, (self.model.input_width, self.model.input_height))
        print("conf_threshold", conf_threshold)
        image_w_detections = self.model.detect_objects(image, conf_threshold)
        image_w_detections = cv2.resize(image_w_detections, (500, 500))
        # add round trip time to image
        # Get text size to position it in lower right
        # Split text into two lines and render separately
        text1 = "Round trip time:"
        text2 = f"{round_trip_time * 1000:>6.1f} msec"
        font_scale = 0.8
        thickness = 2

        # Get text sizes
        (text1_width, text1_height), _ = cv2.getTextSize(
            text1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        (text2_width, text2_height), _ = cv2.getTextSize(
            text2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Position text in bottom right, with text2 below text1
        margin = 10
        text1_x = image_w_detections.shape[1] - text1_width - margin
        text1_y = image_w_detections.shape[0] - text1_height - margin - text2_height
        text2_x = image_w_detections.shape[1] - text2_width - margin
        text2_y = image_w_detections.shape[0] - margin

        # Draw both lines of text
        cv2.putText(
            image_w_detections,
            text1,
            (text1_x, text1_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0, 128),
            thickness,
        )
        cv2.putText(
            image_w_detections,
            text2,
            (text2_x, text2_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0, 128),
            thickness,
        )

        return image_w_detections

    @modal.asgi_app()
    def endpoint(self):
        import asyncio
        import json

        import cv2
        import numpy as np
        from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse

        web_app = FastAPI()

        @web_app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str) -> None:
            await websocket.accept()
            self.websockets[client_id] = websocket
            self.delay_msec[client_id] = 0.0
            self.conf_threshold[client_id] = 0.15
            self.last_frame_time[client_id] = None

            if len(self.websockets) == 1:
                self.frame_processor_task = asyncio.create_task(self.handle_queue())

            print("WebSocket connection accepted")
            while True:
                try:
                    data = await websocket.receive_bytes()

                    if data and isinstance(data, bytes):
                        # Convert bytes to numpy array
                        nparr = np.frombuffer(data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        self.latest_frame[client_id] = Frame(
                            client_id=client_id, image=img
                        )
                        self.frame_queue.put_nowait(
                            Frame(client_id=client_id, image=img)
                        )

                        continue

                except Exception as e:
                    print(f"Websocket handling exception for client {client_id}: {e}")
                    if isinstance(e, WebSocketDisconnect):
                        print(f"Websocket {client_id} disconnected")
                        await websocket.close()
                        self.websockets.pop(client_id)
                        if len(self.websockets) == 0:
                            print(
                                "No websocket connections left, cancelling frame processor task"
                            )
                            self.frame_processor_task.cancel()
                        return

                try:
                    data = await websocket.receive_text()
                    if data and isinstance(data, str):
                        try:
                            # try parsing as json
                            json_data = json.loads(data)

                            print("Received json data:", json_data)

                            if json_data["type"] == "confidenceThresh":
                                print(
                                    "Setting confidence threshold to",
                                    json_data["value"],
                                )
                                self.conf_threshold[client_id] = json_data["value"]
                                continue
                            if json_data["type"] == "delay":
                                print("Setting delay to", json_data["value"])
                                self.delay_msec[client_id] = json_data["value"]
                                continue
                        except Exception as e:
                            print("Failed to parse websocket message as json", e)
                except Exception as e:
                    print(f"Websocket handling exception for client {client_id}: {e}")
                    if isinstance(e, WebSocketDisconnect):
                        print(f"Websocket {client_id} disconnected")
                        await websocket.close()
                        self.websockets.pop(client_id)
                        if len(self.websockets) == 0:
                            print(
                                "No websocket connections left, cancelling frame processor task"
                            )
                            self.frame_processor_task.cancel()
                        return

        @web_app.get("/")
        async def get(request: Request):
            html = open("/assets/index.html").read()
            base_url = str(request.url).replace("http", "ws")
            html = html.replace("url-placeholder", f"{base_url}ws")
            return HTMLResponse(html)

        return web_app
