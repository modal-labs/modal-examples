from pathlib import Path

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
    .run_commands(
        "pip install --upgrade pip",
    )
    .pip_install(
        "fastapi[standard]==0.115.4",
        "gradio~=5.7.1",
        "fastrtc",
        "opencv-python",
        "tensorrt",
        "torch",
        "onnxruntime-gpu",
    )
    .add_local_dir(this_folder, remote_path="/assets")
)

app = modal.App(
    "fastrtc-yolo-demo",
    image=image,
)

MAX_CONCURRENT_INPUTS = 10


@app.cls(
    gpu="A100",
    image=image,
    # gradio requires sticky sessions
    # so we limit the number of concurrent containers to 1
    # and allow it to scale to 100 concurrent inputs
    min_containers=1,  # let's keep it hot so it's more fun to share
    max_containers=1,
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
class YoloWebRTCApp:
    @modal.enter()
    def load_model(self):
        import time

        import cv2
        import numpy as np
        import onnxruntime
        from huggingface_hub import hf_hub_download

        self.last_frame_time = None

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

    @modal.asgi_app()
    def ui(self):
        import time

        import cv2
        import gradio as gr
        import numpy as np
        from fastapi import FastAPI
        from fastrtc import Stream, VideoStreamHandler
        from gradio import mount_gradio_app

        def detection(image, conf_threshold=0.15, delay_msec=0):
            now = time.time()
            if self.last_frame_time is None:
                round_trip_time = np.nan
            else:
                round_trip_time = now - self.last_frame_time
            self.last_frame_time = now

            time.sleep(delay_msec / 1000)

            print(f"Image shape: {image.shape}")
            image = cv2.resize(image, (self.model.input_width, self.model.input_height))
            print("conf_threshold", conf_threshold)
            new_image = self.model.detect_objects(image, conf_threshold)
            new_image = cv2.resize(new_image, (500, 500))
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
            text1_x = new_image.shape[1] - text1_width - margin
            text1_y = new_image.shape[0] - text1_height - margin - text2_height
            text2_x = new_image.shape[1] - text2_width - margin
            text2_y = new_image.shape[0] - margin

            # Draw both lines of text
            cv2.putText(
                new_image,
                text1,
                (text1_x, text1_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0, 128),
                thickness,
            )
            cv2.putText(
                new_image,
                text2,
                (text2_x, text2_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0, 128),
                thickness,
            )

            return new_image

        with gr.Blocks() as blocks:
            gr.HTML(
                """
            <h1 style='text-align: center'>
            Real-Time Object Detection with YOLO, Modal, and FastRTC
            </h1>
            """
            )
            with gr.Column():
                conf_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    value=0.15,
                    label="Confidence Threshold",
                    render=False,
                )
                delay_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=0,
                    label="Delay (ms)",
                    render=False,
                )
                Stream(
                    handler=VideoStreamHandler(detection, skip_frames=True),
                    modality="video",
                    mode="send-receive",
                    rtc_configuration={
                        "iceServers": [{"url": "stun:stun.l.google.com:19302"}]
                    },
                    ui_args={
                        "title": "Press Record to Start Object Detection",
                    },
                    track_constraints={
                        "width": {"exact": 640},
                        "height": {"exact": 480},
                        "frameRate": {"min": 30},
                        "facingMode": {"ideal": "environment"},
                    },
                    additional_inputs=[conf_slider, delay_slider],
                    concurrency_limit=MAX_CONCURRENT_INPUTS,
                )

        return mount_gradio_app(app=FastAPI(), blocks=blocks, path="/")
