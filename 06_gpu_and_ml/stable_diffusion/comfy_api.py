# ---
# lambda-test: false
# ---
#
# # Make API calls to a ComfyUI server
#
# This example shows you how to execute ComfyUI workflows via ComfyUI's API.
#
# ![example comfyui workspace](./comfyui-hero.png)
import json
import os
import pathlib
import urllib
import uuid
from typing import Optional

import modal

from comfy_ui import stub as comfyui_stub

stub = modal.Stub(name="example-comfy-api")
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "websocket-client==1.6.4"
)

# This workflow JSON has been exported by running `comfy_ui.py` and downloading the JSON
# using the web UI.
comfyui_workflow_data_path = assets_path = (
    pathlib.Path(__file__).parent / "comfy_ui_workflow.json"
)


def fetch_image(
    filename: str, subfolder: str, folder_type: str, server_address: str
) -> bytes:
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(
        "https://{}/view?{}".format(server_address, url_values)
    ) as response:
        return response.read()


def fetch_history(prompt_id: str, server_address: str) -> Optional[dict]:
    with urllib.request.urlopen(
        f"https://{server_address}/history/{prompt_id}"
    ) as response:
        output = json.loads(response.read())
    return output[prompt_id].get("outputs") if prompt_id in output else None


def run_workflow(
    ws, prompt: str, server_address: str, client_id: str
) -> list[bytes]:
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(
        "https://{}/prompt".format(server_address), data=data
    )
    response_data = json.loads(urllib.request.urlopen(req).read())
    prompt_id = response_data["prompt_id"]
    output_images = {}

    while True:
        out = ws.recv()
        if isinstance(out, str):
            print(f"recieved str msg from websocket. ws msg: {out}")
            try:
                message = json.loads(out)
            except json.JSONDecodeError:
                print(f"expected valid JSON but got: {out}")
                raise
            print(f"received msg from ws: {message}")
            if message["type"] == "executing":
                data = message["data"]
                if data["node"] is None and data["prompt_id"] == prompt_id:
                    break  # Execution is done!
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
    ws_address = f"wss://{server_address}/ws?clientId={client_id}"
    print(f"Connecting to websocket at {ws_address} ...")
    ws.connect(ws_address)
    print(f"Connected at {ws_address}. Running workflow via API")
    images = run_workflow(ws, workflow_data, server_address, client_id)
    image_list = []
    for node_id in images:
        for image_data in images[node_id]:
            image_list.append(image_data)
    return image_list


@stub.local_entrypoint()
def main() -> None:
    workflow_data = json.loads(comfyui_workflow_data_path.read_text())
    prompt = "bag of wooden blocks"

    # Run the ComfyUI server app and make an API call to it.
    # The ComfyUI server app will shutdown on exit of this context manager.
    with comfyui_stub.run(
        show_progress=False,  # hide server app's modal progress logs
        stdout=open(os.devnull, "w"),  # hide server app's application logs
    ) as comfyui_app:
        print(f"{comfyui_app.app_id=}")
        comfyui_url = comfyui_app.web.web_url

        server_address = comfyui_url.split("://")[1]  # strip protocol

        image_list = query_comfy_via_api.remote(
            workflow_data=workflow_data,
            prompt=prompt,
            server_address=server_address,
            client_id=str(uuid.uuid4()),
        )

    for i, img_bytes in enumerate(image_list):
        filename = f"comfyui_{i}.png"
        with open(filename, "wb") as f:
            f.write(img_bytes)
            f.close()
        print(f"saved '{filename}'")
