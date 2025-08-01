"""
This script implements a GPU-powered YOLO inference server that communicates
with a client over QUIC. It uses NAT hole punching to establish a
direct connection through firewalls.
Networking code adapted from:
https://gist.github.com/aksh-at/e85a5517610a1a2bff35fac41d4c982f
YOLO model code from:
https://github.com/modal-labs/modal-examples/tree/main/07_web_endpoints/webrtc
Usage:
# Start server (rendezvous + YOLO on GPU)
> modal serve quic_yolo_modal.py
# Run client locally
> uv run client.py --url <rendezvous_url> [--fake]  # --fake if no webcam available
"""

import asyncio
import socket
import ssl
import uuid
from pathlib import Path
from typing import Literal

import modal

server_id = str(uuid.uuid4())

# Modal setup
app_name = "modal-quic-yolo"
py_version = "3.12"
tensorrt_ld_path = f"/usr/local/lib/python{py_version}/site-packages/tensorrt_libs"

image = (
    modal.Image.debian_slim(python_version=py_version)  # matching ld path
    # update locale as required by onnx
    .apt_install("locales")
    .run_commands(
        "sed -i '/^#\\s*en_US.UTF-8 UTF-8/ s/^#//' /etc/locale.gen",  # use sed to uncomment
        "locale-gen en_US.UTF-8",  # set locale
        "update-locale LANG=en_US.UTF-8",
    )
    .env({"LD_LIBRARY_PATH": tensorrt_ld_path, "LANG": "en_US.UTF-8"})
    # install system dependencies
    .apt_install("python3-opencv", "ffmpeg")
    # install Python dependencies
    .pip_install(
        "aioquic==1.2.0",
        "cryptography==45.0.4",
        "huggingface-hub[hf_xet]==0.33.0",
        "onnxruntime-gpu==1.21.0",
        "opencv-python==4.11.0.86",
        "tensorrt==10.9.0.34",
        "pynat==0.6.0",
    )
)

cache_vol = modal.Volume.from_name(f"{app_name}-cache", create_if_missing=True)
cache_dir = Path("/cache")
cache = {cache_dir: cache_vol}

app = modal.App(app_name)


with image.imports():
    import cv2
    import numpy as np
    import onnxruntime
    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric.dsa import DSAPrivateKey

    async def get_ext_addr(sock: socket.socket) -> tuple[str, int]:
        from pynat import get_stun_response

        response = get_stun_response(sock, ("stun.ekiga.net", 3478))
        return response["ext_ip"], response["ext_port"]

    def create_cert(key: DSAPrivateKey) -> x509.Certificate:
        import datetime

        from cryptography.hazmat.primitives import hashes
        from cryptography.x509.oid import NameOID

        return (
            x509.CertificateBuilder()
            .subject_name(
                x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, app_name)])
            )
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, app_name)]))
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=1))
            .sign(key, hashes.SHA256())
        )

    class YOLOv10:
        def __init__(self, cache_dir: Path):
            from huggingface_hub import hf_hub_download

            # initialize model
            self.cache_dir = cache_dir
            model_file = hf_hub_download(
                repo_id="onnx-community/yolov10n",
                filename="onnx/model.onnx",
                cache_dir=self.cache_dir,
            )
            self.initialize_model(model_file)

        def initialize_model(self, model_file: Path):
            self.session = onnxruntime.InferenceSession(
                model_file,
                providers=[
                    (
                        "TensorrtExecutionProvider",
                        {
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": self.cache_dir / "onnx.cache",
                        },
                    ),
                    "CUDAExecutionProvider",
                ],
            )
            # get model info
            self.get_input_details()
            self.get_output_details()

            # class names
            self.class_names = [
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat",
                "traffic light",
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "backpack",
                "umbrella",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
                "bottle",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
                "chair",
                "couch",
                "potted plant",
                "bed",
                "dining table",
                "toilet",
                "tv",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush",
            ]
            self.colors = np.random.default_rng(3).uniform(
                0, 255, size=(len(self.class_names), 3)
            )

        def detect_objects(
            self, image: np.ndarray, conf_threshold: float = 0.3
        ) -> np.ndarray:
            input_tensor = self.prepare_input(image)
            new_image = self.inference(image, input_tensor, conf_threshold)
            return new_image

        def prepare_input(self, image: np.ndarray) -> np.ndarray:
            self.img_height, self.img_width = image.shape[:2]

            input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # resize input image to model input size
            input_img = cv2.resize(input_img, (self.input_width, self.input_height))

            # scale input pixel values to 0 to 1
            input_img = input_img / 255.0
            input_img = input_img.transpose(2, 0, 1)
            input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

            return input_tensor

        def inference(
            self,
            image: np.ndarray,
            input_tensor: np.ndarray,
            conf_threshold: float = 0.3,
        ) -> np.ndarray:
            # set seed to potentially create smoother output in RT setting
            onnxruntime.set_seed(42)
            # start = time.perf_counter()
            outputs = self.session.run(
                self.output_names, {self.input_names[0]: input_tensor}
            )

            # print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
            (
                boxes,
                scores,
                class_ids,
            ) = self.process_output(outputs, conf_threshold)
            return self.draw_detections(image, boxes, scores, class_ids)

        def process_output(
            self, output: np.ndarray, conf_threshold: float = 0.3
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            predictions = np.squeeze(output[0])

            # filter out object confidence scores below threshold
            scores = predictions[:, 4]
            predictions = predictions[scores > conf_threshold, :]
            scores = scores[scores > conf_threshold]

            if len(scores) == 0:
                return [], [], []

            # get the class with the highest confidence
            class_ids = predictions[:, 5].astype(int)

            # get bounding boxes for each object
            boxes = self.extract_boxes(predictions)

            return boxes, scores, class_ids

        def extract_boxes(self, predictions: np.ndarray) -> np.ndarray:
            # extract boxes from predictions
            boxes = predictions[:, :4]

            # scale boxes to original image dimensions
            boxes = self.rescale_boxes(boxes)

            # convert boxes to xyxy format
            # boxes = xywh2xyxy(boxes)

            return boxes

        def rescale_boxes(self, boxes: np.ndarray) -> np.ndarray:
            # rescale boxes to original image dimensions
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
                [self.img_width, self.img_height, self.img_width, self.img_height]
            )
            return boxes

        def draw_detections(
            self,
            image: np.ndarray,
            boxes: np.ndarray,
            scores: np.ndarray,
            class_ids: np.ndarray,
        ) -> np.ndarray:
            det_img = image.copy()

            img_height, img_width = image.shape[:2]
            font_size = min([img_height, img_width]) * 0.0012
            text_thickness = int(min([img_height, img_width]) * 0.004)

            # draw bounding boxes and labels of detections
            for class_id, box, score in zip(class_ids, boxes, scores):
                color = self.colors[class_id]

                self.draw_box(det_img, box, color)  # type: ignore

                label = self.class_names[class_id]
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
            self.output_names = [
                model_outputs[i].name for i in range(len(model_outputs))
            ]

        def draw_box(
            self,
            image: np.ndarray,
            box: np.ndarray,
            color: tuple[int, int, int] = (0, 0, 255),
            thickness: int = 5,
        ) -> np.ndarray:
            x1, y1, x2, y2 = box.astype(int)
            return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        def draw_text(
            self,
            image: np.ndarray,
            text: str,
            box: np.ndarray,
            color: tuple[int, int, int] = (0, 0, 255),
            font_size: float = 0.100,
            text_thickness: int = 5,
            box_thickness: int = 5,
        ) -> np.ndarray:
            x1, y1, _, _ = box.astype(int)
            (tw, th), _ = cv2.getTextSize(
                text=text,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_size,
                thickness=text_thickness,
            )
            x1 = x1 - box_thickness
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

    def get_yolo_model(cache_path: Path) -> YOLOv10:
        import onnxruntime

        onnxruntime.preload_dlls()
        return YOLOv10(cache_path)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes=cache,
    region="us-west-1",
    max_inputs=1,
)
async def yolo_quic_server(
    *,
    rendezvous_url: str,
    target_id: str,
    local_port: int = 5555,
):
    import aiohttp
    import cv2
    import numpy as np
    from aioquic.asyncio import serve
    from aioquic.quic.configuration import QuicConfiguration
    from cryptography.hazmat.primitives.asymmetric import ec

    # discover public mapping via STUN
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", local_port))
    sock.setblocking(False)

    pub_ip, pub_port = await get_ext_addr(sock)
    print(f"[{target_id}] Public tuple: {pub_ip}:{pub_port}")

    # register & wait for the peer's tuple
    async with aiohttp.ClientSession() as session:
        while True:
            resp = await session.post(
                f"{rendezvous_url}/register",
                json={
                    "role": "server",
                    "peer_id": server_id,
                    "target_id": target_id,
                    "ip": pub_ip,
                    "port": pub_port,
                },
            )
            if peer := (await resp.json()).get("peer"):
                peer_ip, peer_port = peer
                break
            await asyncio.sleep(1)
    print(f"[{target_id}] Peer tuple: {peer_ip}:{peer_port}")

    for _ in range(150):  # 15s total
        sock.sendto(b"punch", (peer_ip, peer_port))
        try:
            await asyncio.wait_for(asyncio.get_event_loop().sock_recv(sock, 16), 0.1)
            break
        except asyncio.TimeoutError:
            continue
    else:
        raise RuntimeError(
            f"[{target_id}] NAT hole punching failed – no response from peer"
        )
    print(f"[{target_id}] Punched {pub_ip}:{pub_port} -> {peer_ip}:{peer_port}")

    sock.close()  # close socket, mapping should stay alive

    cfg = QuicConfiguration(
        is_client=False,
        alpn_protocols=["hq-29"],
        verify_mode=ssl.CERT_NONE,
    )
    cfg.private_key = ec.generate_private_key(ec.SECP256R1())
    cfg.certificate = create_cert(cfg.private_key)

    yolo = get_yolo_model(cache_dir)

    all_done = asyncio.Event()

    async def handle_stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        frame_idx = 0

        try:
            while True:
                try:
                    header = await reader.readexactly(4)  # 4-byte length header
                except asyncio.IncompleteReadError:
                    # Client disconnected abruptly
                    break
                frame_len = int.from_bytes(header, "big")
                if frame_len == 0:  # client finished, stop loop
                    break
                try:
                    data = await reader.readexactly(frame_len)
                except asyncio.IncompleteReadError:
                    break

                # decode JPEG bytes → ndarray
                np_arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    print("Failed to decode frame; skipping")
                    continue

                # run inference
                annotated = yolo.detect_objects(frame)

                # re-encode as JPEG
                ok, buf = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                if not ok:
                    print("JPEG encode failed; skipping frame")
                    continue
                out_bytes = buf.tobytes()

                writer.write(len(out_bytes).to_bytes(4, "big") + out_bytes)
                await writer.drain()

                frame_idx += 1
        finally:
            writer.close()
            all_done.set()
            try:
                await asyncio.wait_for(writer.wait_closed(), timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                pass

    def stream_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        asyncio.create_task(handle_stream(reader, writer))  # run in the background

    server = await serve(
        host="0.0.0.0",
        port=local_port,  # Use the punched port.
        configuration=cfg,
        stream_handler=stream_handler,
    )

    await all_done.wait()
    server.close()
    print(f"[{target_id}] Shutting down server")


@app.cls(image=image, max_containers=1)
class Rendezvous:
    peers: dict[str, tuple[str, str, int]]
    spawned_clients: set[str]

    @modal.enter()
    def _enter(self):
        self.peers = {}
        self.spawned_clients = set()

    @modal.asgi_app()
    def app(self):
        from fastapi import FastAPI, Request
        from pydantic import BaseModel

        api = FastAPI()

        class RegisterRequest(BaseModel):
            role: Literal["server", "client"]
            peer_id: str
            target_id: str | None
            ip: str
            port: int

        @api.post("/register")
        async def register(req: RegisterRequest, request: Request):
            self.peers[req.peer_id] = (req.role, req.ip, req.port)

            if req.role == "client" and req.peer_id not in self.spawned_clients:
                base_url = str(request.base_url).rstrip("/")
                yolo_quic_server.spawn(rendezvous_url=base_url, target_id=req.peer_id)
                self.spawned_clients.add(req.peer_id)

            if req.role == "server":
                for peer_id, (role, ip, port) in self.peers.items():
                    if role == "client" and peer_id == req.target_id:
                        return {"peer": (ip, port)}
            else:
                for peer_id, (role, ip, port) in self.peers.items():
                    if role == "server":
                        return {"peer": (ip, port)}
            return {"peer": None}

        return api
