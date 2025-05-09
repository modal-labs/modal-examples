# ---
# lambda-test: false
# ---
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

this_dir = Path(__file__).parent.resolve()


class YOLOv10:
    def __init__(self, cache_dir):
        from huggingface_hub import hf_hub_download

        # Initialize model
        self.cache_dir = cache_dir
        print(f"Initializing YOLO model from {self.cache_dir}")
        model_file = hf_hub_download(
            repo_id="onnx-community/yolov10n",
            filename="onnx/model.onnx",
            cache_dir=self.cache_dir,
        )
        self.initialize_model(model_file)
        print("YOLO model initialized")

    def initialize_model(self, model_file):
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
        # Get model info
        self.get_input_details()
        self.get_output_details()

        # get class names
        with open(this_dir / "yolo_classes.txt", "r") as f:
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
            [self.img_width, self.img_height, self.img_width, self.img_height]
        )
        return boxes

    def draw_detections(
        self, image, boxes, scores, class_ids, draw_scores=True, mask_alpha=0.4
    ):
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        font_size = min([img_height, img_width]) * 0.0012
        text_thickness = int(min([img_height, img_width]) * 0.004)

        # Draw bounding boxes and labels of detections
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
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

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
        x1, y1, x2, y2 = box.astype(int)
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
