import json
import os
import pathlib
import urllib
import uuid

import modal

from comfy_ui import stub as comfyui_stub

stub = modal.Stub(name="example-comfy-api")
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "websocket-client"
)

# This workflow JSON has been exported by running `comfy_ui.py` and downloading the JSON
# using the web UI.
comfyui_workflow_data_path = assets_path = (
    pathlib.Path(__file__).parent / "comfy_ui_workflow.json"
)


def fetch_image(filename, subfolder, folder_type, server_address):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(
        "https://{}/view?{}".format(server_address, url_values)
    ) as response:
        return response.read()


def fetch_history(prompt_id, server_address) -> dict:
    with urllib.request.urlopen(
        "https://{}/history/{}".format(server_address, prompt_id)
    ) as response:
        output = json.loads(response.read())
        return output[prompt_id]["outputs"]


def run_workflow(ws, prompt, server_address, client_id):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(
        "https://{}/prompt".format(server_address), data=data
    )
    prompt_id = json.loads(urllib.request.urlopen(req).read())["prompt_id"]
    output_images = {}

    while True:
        # believe we need this check to make sure the job is finished running before checking history
        out = ws.recv()
        print("recieved msg from websocket")
        if isinstance(out, str):
            try:
                message = json.loads(out)
            except json.JSONDecodeError:
                print(f"expected valid JSON but got: {out}")
                raise
            # if message["data"]["sid"]:
            # break  # execution is done
            print(f"ws msg: {message}")
            if message["type"] == "executing":
                data = message["data"]
                if data["node"] is None and data["prompt_id"] == prompt_id:
                    break  # Execution is done
        else:
            continue  # previews are binary data
    history = fetch_history(prompt_id, server_address)
    for node_id in history:
        node_output = history[node_id]
        if "images" in node_output:
            images_output = []
            for image in node_output["images"]:
                image_data = fetch_image(
                    image["filename"],
                    image["subfolder"],
                    image["type"],
                    server_address,
                )
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images


@stub.function(image=image)
def query_comfy_via_api(
    workflow_data: dict, prompt: str, server_address: str, client_id: str
):
    import websocket

    # Modify workflow to use requested prompt.
    workflow_data["2"]["inputs"]["text"] = prompt
    ws = websocket.WebSocket()
    ws.connect("wss://{}/ws?clientId={}".format(server_address, client_id))
    images = run_workflow(ws, workflow_data, server_address, client_id)
    image_list = []
    for node_id in images:
        for image_data in images[node_id]:
            image_list.append(image_data)
    return image_list


@stub.local_entrypoint()
def main():
    workflow_data = json.loads(comfyui_workflow_data_path.read_text())
    prompt = "bag of wooden blocks"

    with comfyui_stub.run(show_progress=False) as comfyui_app:
        print(f"{comfyui_app.app_id=}")
        comfyui_url = comfyui_app.web.web_url

        server_address = comfyui_url.split("://")[1]  # strip protocol
        client_id = str(uuid.uuid4())

        image_list = query_comfy_via_api.remote(
            workflow_data, prompt, server_address, client_id
        )

        for i, img_bytes in enumerate(image_list):
            with open(f"comfyui_{i}.png", "wb") as f:
                f.write(img_bytes)
                f.close()
        os.system(f"open comfyui_{i}.png")  # open last image
