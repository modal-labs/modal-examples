import numpy as np
import cv2

class_names = [
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

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices, :]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    # det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # Draw bounding boxes and labels of detections
    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]

        draw_box(det_img, box, color)  # type: ignore

        label = class_names[class_id]
        caption = f"{label} {int(score * 100)}%"
        draw_text(det_img, caption, box, color, font_size, text_thickness)  # type: ignore

    return det_img


def draw_box(
    image: np.ndarray,
    box: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(
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


def draw_masks(
    image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float = 0.3
) -> np.ndarray:
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)  # type: ignore

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

import modal

web_image = (
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
        "onnxruntime-gpu"
    )
)

app = modal.App(
    "fastrtc-yolo-demo",
    image=web_image,
)


@app.cls(
    gpu="A100",
    image=web_image,
    min_containers=1,
    scaledown_window=60 * 20,
    # gradio requires sticky sessions
    # so we limit the number of concurrent containers to 1
    # and allow it to scale to 100 concurrent inputs
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
class YoloWebRTCApp:
    
    @modal.enter()
    def load_model(self):

        import time
        import numpy as np

        from huggingface_hub import hf_hub_download
        import cv2
        import onnxruntime

        onnxruntime.preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None)


        class YOLOv10:
            def __init__(self, path):
                # Initialize model
                model_file = hf_hub_download(
                    repo_id="onnx-community/yolov10n", filename="onnx/model.onnx"
                )
                self.initialize_model(model_file)

            def __call__(self, image):
                return self.detect_objects(image)

            def initialize_model(self, path):
                self.session = onnxruntime.InferenceSession(
                    path, providers=onnxruntime.get_available_providers()
                )
                # Get model info
                self.get_input_details()
                self.get_output_details()

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
                return draw_detections(image, boxes, scores, class_ids, mask_alpha)

            def get_input_details(self):
                model_inputs = self.session.get_inputs()
                self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

                self.input_shape = model_inputs[0].shape
                self.input_height = self.input_shape[2]
                self.input_width = self.input_shape[3]

            def get_output_details(self):
                model_outputs = self.session.get_outputs()
                self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        
    
        self.model = YOLOv10("yolov10m.pt")

    

    @modal.asgi_app()
    def ui(self):

        import gradio as gr
        from gradio import mount_gradio_app
        from fastapi import FastAPI
        from fastrtc import VideoStreamHandler, Stream


        def detection(image, conf_threshold = 0.5):
            image = cv2.resize(image, (self.model.input_width, self.model.input_height))
            print("conf_threshold", conf_threshold)
            new_image = self.model.detect_objects(image, conf_threshold)
            return cv2.resize(new_image, (500, 500))

        with gr.Blocks() as blocks:
            gr.HTML(
            """
            <h1 style='text-align: center'>
            Real-Time Object Detection with YOLO, Modal, and FastRTC
            </h1>
            """
            )
            with gr.Column():
                slider = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.5, label="Confidence Threshold", render=False)
                stream = Stream(
                    handler=VideoStreamHandler(detection, skip_frames=True, fps = 30.0),
                    modality="video",
                    mode="send-receive",
                    rtc_configuration={
                        "iceServers": [{"url": "stun:stun.l.google.com:19302"}]
                    },
                    # additional_inputs=[0.3],
                    ui_args={
                        "pulse_color": "rgb(255, 255, 255)",
                        "icon_button_color": "rgb(255, 255, 255)",
                        "title": "Press Record to Start Object Detection",
                    },
                    additional_inputs=[slider],
                )
                
            
        return mount_gradio_app(app=FastAPI(), blocks=blocks, path="/")