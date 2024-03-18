# ---
# lambda-test: false
# ---
#
# # Make API calls to a ComfyUI server
#
# This example shows you how to execute ComfyUI JSON-defined workflows via ComfyUI's API.
# It also provides a helper function `get_python_workflow`` that maps a JSON-defined workflow into Python objects.
# ![example comfyui workspace](./comfyui-hero.png)
import json
import os
import pathlib
import urllib
import uuid

import modal

comfyui_commit_sha = "a38b9b3ac152fb5679dad03813a93c09e0a4d15e"

# This workflow JSON has been exported by running `comfy_ui.py` and downloading the JSON
# using the web UI.
comfyui_workflow_data_path = assets_path = (
    pathlib.Path(__file__).parent / "workflow_api.json"
)

stub = modal.Stub(name="example-comfy-api")
from comfy_ui import image


def fetch_image(
    filename: str, subfolder: str, folder_type: str, server_address: str
) -> bytes:
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(
        "https://{}/view?{}".format(server_address, url_values)
    ) as response:
        return response.read()


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

    # Fetch workflow execution history, which contains references to our completed images.
    with urllib.request.urlopen(
        f"https://{server_address}/history/{prompt_id}"
    ) as response:
        output = json.loads(response.read())
    history = output[prompt_id].get("outputs") if prompt_id in output else None
    if not history:
        raise RuntimeError(
            f"Unexpected missing ComfyUI history for {prompt_id}"
        )
    for node_id in history:
        node_output = history[node_id]
        if "images" in node_output:
            images_output = []
            for image in node_output["images"]:
                image_data = fetch_image(
                    filename=image["filename"],
                    subfolder=image["subfolder"],
                    folder_type=image["type"],
                    server_address=server_address,
                )
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images


# Execute a run of a JSON-defined workflow on a remote ComfyUI server
# This is adapted from the ComfyUI script examples: https://github.com/comfyanonymous/ComfyUI/blob/master/script_examples/websockets_api_example.py
# A better way to execute a workflow programmatically is to convert the JSON to Python code using convert_workflow_to_python
# Then importing that generated code into a Modal endpoint; see serve_workflow.py
@stub.function(image=image)
def query_comfy_via_api(workflow_data: dict, prompt: str, server_address: str):
    import websocket

    # Modify workflow to use requested prompt.
    workflow_data["2"]["inputs"]["text"] = prompt

    # Make a websocket connection to the ComfyUI server. The server will
    # will stream workflow execution updates over this websocket.
    ws = websocket.WebSocket()
    client_id = str(uuid.uuid4())
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


@stub.function(
    image=image,
    gpu="any",
    mounts=[
        modal.Mount.from_local_file(
            comfyui_workflow_data_path, "/root/workflow_api.json"
        )
    ],
)
def convert_workflow_to_python():
    import subprocess

    process = subprocess.Popen(
        ["python", "./ComfyUI-to-Python-Extension/comfyui_to_python.py"]
    )
    process.wait()
    retcode = process.returncode

    if retcode != 0:
        raise RuntimeError(
            f"comfy_api.py exited unexpectedly with code {retcode}"
        )
    else:
        try:
            return pathlib.Path("workflow_api.py").read_text()
        except FileNotFoundError:
            print("Error: File workflow_api.py not found.")


# Generate a Python representation of workflow_api.json using this extension: https://github.com/pydn/ComfyUI-to-Python-Extension
# First, you need to download your workflow_api.json from ComfyUI and save it to this directory.
# Then, this function will generate a Python version to _generated_workflow_api.py, which you'll reference in workflow_api.py.
@stub.local_entrypoint()
def get_python_workflow():
    workflow_text = convert_workflow_to_python.remote()
    filename = "_generated_workflow_api.py"
    pathlib.Path(filename).write_text(workflow_text)
    print(f"saved '{filename}'")


@stub.local_entrypoint()
def main(prompt: str = "bag of wooden blocks") -> None:
    workflow_data = json.loads(comfyui_workflow_data_path.read_text())

    # Run the ComfyUI server app and make an API call to it.
    # The ComfyUI server app will shutdown on exit of this context manager.
    from comfy_ui import stub as comfyui_stub

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
        )

    for i, img_bytes in enumerate(image_list):
        filename = f"comfyui_{i}.png"
        with open(filename, "wb") as f:
            f.write(img_bytes)
            f.close()
        print(f"saved '{filename}'")
