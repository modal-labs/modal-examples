# # Memory snapshots for Stable Diffusion XL
#
# We will load Stable Diffusion XL 1.0 in CPU memory then take a memory snapshot
# to improve cold boot times.

import time
import modal


MODEL_ID: str = "stabilityai/stable-diffusion-xl-base-1.0"
CACHE_PATH: str = "/vol/cache"


image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "diffusers",
        "accelerate",
        "huggingface_hub",
        "hf_transfer",
        "invisible_watermark",
        "safetensors",
    )
    .apt_install("ffmpeg", "libsm6", "libxext6")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
stub = modal.Stub("sdxl-2", image=image)

## Create a modal class with memory snapshots enabled
# 
# `checkpointing_enabled=True` creates a Modal class with memory snapshots enabled.
# When this is set to `True` only imports are snapshotted. Use it in combination with
# `@enter(checkpoint=True)` to add load model weights in CPU memory and the snapshot.
# You can transfer weights to a GPU in `@enter(checkpoint=False)`. All methods decorated
# with `@enter(checkpoint=True)` are only executed during the snapshotting stage.


@stub.cls(gpu=modal.gpu.A10G(), checkpointing_enabled=True)
class SDXL:

    @modal.build()
    def build(self):
        import diffusers
        import transformers

        pipe = diffusers.DiffusionPipeline.from_pretrained(
            MODEL_ID, use_safetensors=True
        )
        transformers.utils.move_cache()
        pipe.save_pretrained(CACHE_PATH)

    @modal.enter(checkpoint=True)
    def load(self):
        import diffusers

        self.pipe = diffusers.DiffusionPipeline.from_pretrained(
            CACHE_PATH, use_safetensors=True
        )
        self.pipe.to("cpu")

    @modal.enter(checkpoint=False)
    def setup(self):
        self.pipe.to("cuda")


    @modal.method()
    def run(self) -> str:
        prompt = "blocks made of steel floating in the clouds, 8k, cinematic, glass, Unreal Engine"
        return self.pipe(prompt=prompt, num_inference_steps=1).images[0]



if __name__ == "__main__":
    sdxl = modal.Cls.lookup("sdxl-2", "SDXL")

    start = time.time()
    sdxl().run.remote()
    print(f"e2e took => {time.time() - start:3f}s")
