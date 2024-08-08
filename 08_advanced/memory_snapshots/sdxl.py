# # Memory snapshots for Stable Diffusion XL (SDXL)
#
# This examples shows how memory snapshots can improve cold boot times when using
# GPUs and large models, in this case SDXL.
#
# We will first load [StabilityAI's `stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) model
# in CPU memory then take a memory snapshot. Whenever this Modal App scales up, it will
# start from the snapshot instead of loading the original model files from the
# file system (about 7GB of model weights alone). This cuts ~5s from cold boot
# times (from around 17s to 12s).

import time

import modal

MODEL_ID: str = "stabilityai/stable-diffusion-xl-base-1.0"
CACHE_PATH: str = "/vol/cache"


image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch==2.2.1",
        "transformers==4.38.2",
        "diffusers==0.26.3",
        "hf_transfer==0.1.6",
        # "invisible_watermark",
        "safetensors==0.4.2",
    )
    .apt_install("ffmpeg", "libsm6", "libxext6")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
stub = modal.Stub("sdxl", image=image)

## Create a modal class with memory snapshots enabled
#
# `enable_memory_snapshot=True` creates a Modal class with memory snapshots enabled.
# When this is set to `True` only imports are snapshotted. Use it in combination with
# `@enter(snap=True)` to add load model weights in CPU memory and the snapshot.
# You can transfer weights to a GPU in `@enter(snap=False)`. All methods decorated
# with `@enter(snap=True)` are only executed during the snapshotting stage.


@stub.cls(gpu=modal.gpu.A10G(), enable_memory_snapshot=True)
class SDXL:

    @modal.build()
    def build(self):
        import diffusers
        import transformers

        pipe = diffusers.DiffusionPipeline.from_pretrained(
            MODEL_ID, use_safetensors=True,
            low_cpu_mem_usage=False,
        )
        transformers.utils.move_cache()
        pipe.save_pretrained(CACHE_PATH)

    @modal.enter(snap=True)
    def load_model(self):
        import diffusers

        self.pipe = diffusers.DiffusionPipeline.from_pretrained(
            CACHE_PATH, use_safetensors=True,
            low_cpu_mem_usage=False,
        )
        self.pipe.to("cpu")

    @modal.enter(snap=False)
    def move_to_gpu(self):
        self.pipe.to("cuda")

    @modal.method()
    def run(self) -> str:
        prompt = "blocks made of steel floating in the clouds, 8k, cinematic, glass, Unreal Engine"
        return self.pipe(prompt=prompt, num_inference_steps=1).images[0]



if __name__ == "__main__":
    sdxl = modal.Cls.lookup("sdxl", "SDXL")

    start = time.time()
    sdxl().run.remote()
    print(f"e2e took => {time.time() - start:3f}s")
