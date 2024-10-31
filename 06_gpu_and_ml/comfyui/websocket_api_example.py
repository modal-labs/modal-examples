# This is an example that uses the websockets api and the SaveImageWebsocket node to get images directly without
# them being saved to disk

import json
import urllib.parse
import urllib.request
import uuid

import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)

server_address = "modal-labs--example-comfyui-ui-dev.modal.run"
client_id = str(uuid.uuid4())


def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(
        "https://{}/prompt".format(server_address), data=data
    )
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(
        "https://{}/view?{}".format(server_address, url_values)
    ) as response:
        return response.read()


def get_history(prompt_id):
    with urllib.request.urlopen(
        "https://{}/history/{}".format(server_address, prompt_id)
    ) as response:
        return json.loads(response.read())


def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)["prompt_id"]
    output_images = {}
    current_node = ""
    while True:
        out = ws.recv()
        if isinstance(out, str):
            print(out)
            message = json.loads(out)
            if message["type"] == "executing":
                data = message["data"]
                if data["prompt_id"] == prompt_id:
                    if data["node"] is None:
                        break  # Execution is done
                    else:
                        current_node = data["node"]
        else:
            print(type(out))
            # Save the bytes object to a local file
            output_filename = f"{uuid.uuid4().hex}.jpeg"
            with open(output_filename, "wb") as f:
                f.write(out[8:])
            if current_node == "save_image_websocket_node":
                images_output = output_images.get(current_node, [])
                images_output.append(out[8:])
                output_images[current_node] = images_output

    return output_images


prompt_text = """
{
  "55": {
    "inputs": {
      "seed": 1106939166727692,
      "steps": 20,
      "cfg": 7.5,
      "sampler_name": "dpmpp_sde",
      "scheduler": "karras",
      "denoise": 1,
      "preview_method": "auto",
      "vae_decode": "true",
      "model": [
        "110",
        0
      ],
      "positive": [
        "110",
        1
      ],
      "negative": [
        "110",
        2
      ],
      "latent_image": [
        "110",
        3
      ],
      "optional_vae": [
        "110",
        4
      ]
    },
    "class_type": "KSampler (Efficient)",
    "_meta": {
      "title": "KSampler (Efficient)"
    }
  },
  "110": {
    "inputs": {
      "ckpt_name": "Realistic_Vision_V1.4.safetensors",
      "vae_name": "blessed2.vae.pt",
      "clip_skip": -2,
      "lora_name": "None",
      "lora_model_strength": 1,
      "lora_clip_strength": 1,
      "positive": "masterpiece, best quality, movie still, 1girl, floating in the sky, cloud girl, cloud, (close-up:1.1), bright, happy, fun, soft lighting, closeup",
      "negative": "embedding:EasyNegative.pt, embedding:bad-artist-anime.pt, lowres, low quality, worst quality, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, out of frame, ugly, extra limbs, bad anatomy, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, too many fingers, long neck, kid, teenage, badhandv4, EasyNegative, child, kid, teenage",
      "token_normalization": "length+mean",
      "weight_interpretation": "A1111",
      "empty_latent_width": 768,
      "empty_latent_height": 768,
      "batch_size": 1
    },
    "class_type": "Efficient Loader",
    "_meta": {
      "title": "Efficient Loader"
    }
  }
}
"""

prompt = json.loads(prompt_text)

ws = websocket.WebSocket()
ws_url = "wss://{}/ws?clientId={}".format(server_address, client_id)
print(ws_url)
ws.connect(ws_url)
print("connected")
images = get_images(ws, prompt)
print("done")
ws.close()  # for in case this example is used in an environment where it will be repeatedly called, like in a Gradio app. otherwise, you'll randomly receive connection timeouts
# Commented out code to display the output images:

# for node_id in images:
#     for image_data in images[node_id]:
#         from PIL import Image
#         import io
#         image = Image.open(io.BytesIO(image_data))
#         image.show()
