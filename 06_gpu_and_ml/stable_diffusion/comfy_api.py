import modal
import uuid

stub = modal.Stub(name="example-comfy-api")
image = modal.Image.debian_slim(python_version="3.10").pip_install("websocket-client")

with image.imports():
    import json
    import urllib
    import websocket

def queue_prompt(prompt, server_address, client_id):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("https://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type, server_address):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("https://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id, server_address):
    with urllib.request.urlopen("https://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt, server_address, client_id):
    prompt_id = queue_prompt(prompt, server_address, client_id)['prompt_id']
    output_images = {}
    while True:
        # believe we need this check to make sure the job is finished running before checking history
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['data']['sid']:
                break # execution is done
        else:
            continue #previews are binary data
    history = get_history(prompt_id, server_address)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'], server_address)
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images

@stub.function(image=image)
def query_comfy_via_api(prompt_text, server_address, client_id):
    prompt = json.loads(prompt_text)
    prompt["2"]["inputs"]["text"] = "bag of wooden blocks"
    ws = websocket.WebSocket()
    ws.connect("wss://{}/ws?clientId={}".format(server_address, client_id))
    images = get_images(ws, prompt, server_address, client_id)
    image_list = []
    for node_id in images:
        for image_data in images[node_id]:
            image_list.append(image_data)
    
    return image_list
            

@stub.local_entrypoint()
def main():
    prompt_text = """
    {
    "1": {
        "inputs": {
        "ckpt_name": "dreamlike-photoreal-2.0.safetensors"
        },
        "class_type": "CheckpointLoaderSimple",
        "_meta": {
        "title": "Load Checkpoint"
        }
    },
    "2": {
        "inputs": {
        "text": "a bag of wooden blocks",
        "clip": [
            "1",
            1
        ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
        "title": "CLIP Text Encode (Positive)"
        }
    },
    "3": {
        "inputs": {
        "text": "bag of noodles",
        "clip": [
            "1",
            1
        ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
        "title": "CLIP Text Encode (Negative)"
        }
    },
    "4": {
        "inputs": {
        "seed": 350088449706888,
        "steps": 12,
        "cfg": 8,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 1,
        "model": [
            "1",
            0
        ],
        "positive": [
            "2",
            0
        ],
        "negative": [
            "3",
            0
        ],
        "latent_image": [
            "5",
            0
        ]
        },
        "class_type": "KSampler",
        "_meta": {
        "title": "KSampler"
        }
    },
    "5": {
        "inputs": {
        "width": 512,
        "height": 512,
        "batch_size": 1
        },
        "class_type": "EmptyLatentImage",
        "_meta": {
        "title": "Empty Latent Image"
        }
    },
    "6": {
        "inputs": {
        "samples": [
            "8",
            0
        ],
        "vae": [
            "1",
            2
        ]
        },
        "class_type": "VAEDecode",
        "_meta": {
        "title": "VAE Decode"
        }
    },
    "7": {
        "inputs": {
        "images": [
            "6",
            0
        ]
        },
        "class_type": "PreviewImage",
        "_meta": {
        "title": "Preview Image"
        }
    },
    "8": {
        "inputs": {
        "add_noise": "enable",
        "noise_seed": 350088449706888,
        "steps": 30,
        "cfg": 8,
        "sampler_name": "euler",
        "scheduler": "karras",
        "start_at_step": 12,
        "end_at_step": 10000,
        "return_with_leftover_noise": "disable",
        "model": [
            "1",
            0
        ],
        "positive": [
            "2",
            0
        ],
        "negative": [
            "3",
            0
        ],
        "latent_image": [
            "10",
            0
        ]
        },
        "class_type": "KSamplerAdvanced",
        "_meta": {
        "title": "KSampler (Advanced)"
        }
    },
    "10": {
        "inputs": {
        "upscale_method": "nearest-exact",
        "scale_by": 2,
        "samples": [
            "4",
            0
        ]
        },
        "class_type": "LatentUpscaleBy",
        "_meta": {
        "title": "Upscale Latent By"
        }
    },
    "11": {
        "inputs": {
        "samples": [
            "4",
            0
        ],
        "vae": [
            "1",
            2
        ]
        },
        "class_type": "VAEDecode",
        "_meta": {
        "title": "VAE Decode"
        }
    },
    "12": {
        "inputs": {
        "images": [
            "11",
            0
        ]
        },
        "class_type": "PreviewImage",
        "_meta": {
        "title": "Preview Image"
        }
    }
    }
    """
    server_address = "modal-labs--example-comfy-ui-web-dev.modal.run"
    client_id = str(uuid.uuid4())

    image_list = query_comfy_via_api.remote(prompt_text, server_address, client_id)
    
    from PIL import Image
    import io
    for i in image_list:
        image = Image.open(io.BytesIO(i))
        image.show()
