# # Mount S3 bucket and use it for LoRAs
#
# This example shows how to mount an S3 bucket in a Modal app using [`CloudBucketMount`](/docs/reference/modal.CloudBucketMount).
# We will download LoRA adapters from the [HuggingFace Hub](https://huggingface.co/models) into our S3 bucket
# then read from that bucket when doing inference.
#
# ## Basic setup
#
# You will need to have a S3 bucket and AWS credentials to run this example. Refer to the documentation
# for detailed [IAM permissions](/docs/guide/cloud-bucket-mounts#iam-permissions) your credentials will need.
#
# After you are done creating a bucket and configuring IAM settings,
# you now need to create a [Modal Secret](/docs/guide/secrets). Navigate to the "Secrets" tab and
# click on the AWS card, then fill in the fields with the AWS key and secret created
# previously. Name the Secret `s3-bucket-secret`.

import io
import os
from pathlib import Path

from modal import Stub, Image, CloudBucketMount, Secret, build, enter, method

MOUNT_PATH: Path = Path("/mnt/bucket")
LORAS_PATH: Path = MOUNT_PATH / "loras/v0"

image = Image.debian_slim().pip_install(
    "huggingface_hub==0.21.4",
    "transformers==4.38.2",
    "diffusers==0.26.3",
    "peft==0.9.0",
    "accelerate==0.27.2",
)

# Mount your S3 bucket using `CloudBucketMount`. Mounting your bucket in a Stub makes it available
# for all functions from the stub.
stub = Stub(
    image=image,
    volumes={
        MOUNT_PATH: CloudBucketMount(
            "modal-s3mount-test-bucket",
            secret=Secret.from_name("s3-bucket-secret"),
        )
    },
)

with image.imports():
    import diffusers
    import huggingface_hub
    import torch


# `search_loras()` will use the Hub API to search for LoRAs. We limit LoRAs
# to a maximum size to prevent downloading very large model weights. Feel
# free to adapt to what works best for you. This function is expected
# to run locally.
def search_loras(limit: int, max_model_size: int = 800 * 1024 * 1024):
    api = huggingface_hub.HfApi()

    model_ids: list[str] = []
    for model in api.list_models(
        tags=["lora", "base_model:stabilityai/stable-diffusion-xl-base-1.0"],
        library="diffusers",
        sort="downloads",  # sort by most downloaded
    ):
        try:
            model_size = 0
            for file in api.list_files_info(model.id):
                model_size += file.size

        except huggingface_hub.utils.GatedRepoError:
            print(f"gated model ({model.id}); skipping")
            continue

        # Skip models that are larger than file limit.
        if model_size > max_model_size:
            print(f"model {model.id} is too large; skipping")
            continue

        model_ids.append(model.id)
        if len(model_ids) >= limit:
            return model_ids

    return model_ids


# Download LoRA weights to the S3 mount. Downloading files in this mount will automatically
# upload files to S3. We will run this function in parallel using Modal's [`map`](/docs/reference/modal.Function#map).
@stub.function()
def download_lora(repository_id: str) -> str:
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # CloudBucketMounts will report 0 bytes of available space leading to many
    # unnecessary warnings. Patch the method that emits those warnings.
    from huggingface_hub import file_download

    file_download._check_disk_space = lambda x, y: False

    repository_path = LORAS_PATH / repository_id
    huggingface_hub.snapshot_download(
        repository_id,
        local_dir=repository_path.as_posix().replace(".", "_"),
        allow_patterns=["*.safetensors"],
    )

    downloaded_lora = len(list(repository_path.rglob("*.safetensors"))) > 0
    if downloaded_lora:
        return repository_id


# The `StableDiffusionLoRA` loads Stable Diffusion XL 1.0 as a base model. When doing inference,
# it will also load whichever LoRA you specify. It will load these adapters from the same
# S3 bucket you used to download weights into.
@stub.cls(gpu="a10")
class StableDiffusionLoRA:
    pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"

    @build()
    def build(self):
        diffusers.DiffusionPipeline.from_pretrained(
            self.pipe_id, torch_dtype=torch.float16
        )

    @enter()
    def load(self):
        self.pipe = diffusers.DiffusionPipeline.from_pretrained(
            self.pipe_id, torch_dtype=torch.float16
        ).to("cuda")

    @method()
    def run_inference_with_lora(self, lora_id: str = "CiroN2022/toy-face"):
        for file in (LORAS_PATH / lora_id).rglob("*.safetensors"):
            self.pipe.load_lora_weights(lora_id, weight_name=file.name)
            break

        prompt = "toy_face of a hacker with a hoodie"
        lora_scale = 0.9
        image = self.pipe(
            prompt,
            num_inference_steps=30,
            cross_attention_kwargs={"scale": lora_scale},
            generator=torch.manual_seed(0),
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return buffer.getvalue()

# Finally, create a Modal entrypoint for your program. This allows you to run your
# program using `modal run cloud_bucket_mount_loras.py`
@stub.local_entrypoint()
def main(limit: int = 10):

    # Download LoRAs in parallel.
    example_lora = "ashwin-mahadevan/sd-pokemon-model-lora-sdxl"
    lora_model_ids = [example_lora]
    lora_model_ids += search_loras(limit)

    downloaded_loras = []
    for model in download_lora.map(lora_model_ids):
        if model:
            downloaded_loras.append(model)

    print(f"downloaded {len(downloaded_loras)} loras => {downloaded_loras}")

    # Run inference using one of the downloaded LoRAs.
    example_lora = "ashwin-mahadevan/sd-pokemon-model-lora-sdxl"
    byte_stream = StableDiffusionLoRA().run_inference_with_lora.remote(example_lora)
    dir = Path("/tmp/stable-diffusion-xl")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(byte_stream)
