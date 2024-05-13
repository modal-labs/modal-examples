import io
import pathlib

import requests
from PIL import Image

comfyui_workflow_data_path = (
    pathlib.Path(__file__).parent / "workflow-examples" / "inpainting"
)

url = "https://modal-labs--example-comfyui-api-quickstart-backend-dev.modal.run"
workflow = pathlib.Path(
    comfyui_workflow_data_path / "workflow_api.json"
).read_text()
models = pathlib.Path(comfyui_workflow_data_path / "model.json").read_text()
data = {
    "workflow_data": workflow,
    "models": models,
    "text_prompt": "white dog",
    "input_image_url": "https://raw.githubusercontent.com/comfyanonymous/ComfyUI_examples/master/inpaint/yosemite_inpaint_example.png",
}

res = requests.post(url, json=data)
if res.status_code == 200:
    print("Image finished generating!")
    Image.open(io.BytesIO(res.content)).show()
else:
    print("Request failed!")
