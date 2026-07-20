# This examples shows how one can execute a DAG-like workload leveraging heterogenous compute.
# It shows an example image preprocessing pipeline which downloads, converts to tensor on CPU,
# caption image on GPU, and stores tensor-caption pairs in a Modal volume.

import io

import modal

DATASET = "ethz/food101"
SPLIT = "train"
SHARD_SIZE = 32
IMAGE_SIZE = 224
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
HF_CACHE_PATH = "/root/.cache/huggingface"
OUTPUT_PATH = "/data"

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
output = modal.Volume.from_name("captioned-tensors", create_if_missing=True)

app = modal.App("example-captioned-image-dataset")
cpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "datasets==3.2.0",
    "torch==2.5.1",
    "torchvision==0.20.1",
    "pillow==11.0.0",
)
gpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "torch==2.5.1",
    "transformers==4.46.3",
    "pillow==11.0.0",
)
app.image = cpu_image  # set app's default image to cpu_image


@app.function()
def download(shard_id: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=SPLIT, streaming=True)
    shard = ds.skip(shard_id * SHARD_SIZE).take(SHARD_SIZE)

    rows = []
    for row in shard:
        buf = io.BytesIO()
        row["image"].convert("RGB").save(buf, format="JPEG")
        rows.append({"image_bytes": buf.getvalue()})

    print(f"shard #{shard_id} downloaded {len(rows)} images")
    return rows


@app.function()
def to_tensor(rows: list[dict]) -> bytes:
    import torch
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    images = []
    for row in rows:
        img = Image.open(io.BytesIO(row["image_bytes"])).convert("RGB")
        images.append(transform(img))

    pixel_values = torch.stack(images)

    buf = io.BytesIO()
    torch.save({"pixel_values": pixel_values}, buf)

    print(f"transformed {len(images)} images -> {tuple(pixel_values.shape)}")
    return buf.getvalue()


@app.cls(image=gpu_image, gpu="T4", volumes={HF_CACHE_PATH: hf_cache})
class Captioner:
    @modal.enter()
    def load(self):
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.processor = BlipProcessor.from_pretrained(BLIP_MODEL)
        self.model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
        self.model.to("cuda")
        self.model.eval()

    @modal.method()
    def caption(self, rows: list[dict]) -> list[str]:
        import torch
        from PIL import Image

        images = [
            Image.open(io.BytesIO(row["image_bytes"])).convert("RGB") for row in rows
        ]
        inputs = self.processor(images=images, return_tensors="pt").to("cuda")
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=30)

        captions = self.processor.batch_decode(generated, skip_special_tokens=True)
        print(f"captioned {len(captions)} images on GPU")
        return captions


@app.function(volumes={OUTPUT_PATH: output})
def store(
    shard_bytes: bytes,
    captions: list[str],
    shard_id: int,
) -> dict:
    import os

    import torch

    shard = torch.load(io.BytesIO(shard_bytes), weights_only=True)
    shard["captions"] = captions

    path = os.path.join(OUTPUT_PATH, f"shard_{shard_id:04d}.pt")
    torch.save(shard, path)

    num_samples = int(shard["pixel_values"].shape[0])
    print(f"stored shard #{shard_id}, {num_samples} samples -> {path}")
    return {"shard_id": shard_id, "num_samples": num_samples, "path": path}


@app.function()
def preprocess_shard(shard_id: int) -> dict:
    # DAG:
    #           download
    #             /  \
    #     to_tensor   caption
    #             \  /
    #             store

    rows = download.remote(shard_id)

    tensor_call = to_tensor.spawn(rows)
    caption_call = Captioner().caption.spawn(rows)
    tensor_bytes, captions = modal.FunctionCall.gather(tensor_call, caption_call)

    return store.remote(tensor_bytes, captions, shard_id)


@app.local_entrypoint()
def main(num_shards: int = 4):
    results = list(preprocess_shard.map(range(num_shards)))
    total = sum(res["num_samples"] for res in results)
    print(f"{total} samples saved to volume captioned-tensors")
