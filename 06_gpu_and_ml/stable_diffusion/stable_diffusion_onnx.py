# # Stable Diffusion inference using the ONNX Runtime
#
# This example is similar to the example [Stable Diffusion CLI](/docs/guide/ex/stable_diffusion_cli)
# but running inference unsing the [ONNX Runtime](https://onnxruntime.ai/) instead of PyTorch. We still use the
# [diffusers](https://github.com/huggingface/diffusers) package to load models and pipeline like we do in Stable Diffusion, using
# the `OnnxStableDiffusionPipeline` instead of the `StableDiffusionPipeline`. More details
# on how the ONNX runtime works in diffusers in [this article](https://huggingface.co/docs/diffusers/optimization/onnx).
#
# Inference with ONNX is faster by about ~100ms, with throughput of ~950ms / image on a
# A10G GPU. Cold boot times are higher, however; taking about ~18s for the models to
# be loaded into memory.
#
# _Note:_ this is adapted from the article [Accelerating Stable Diffusion Inference with ONNX Runtime](https://medium.com/microsoftazure/accelerating-stable-diffusion-inference-with-onnx-runtime-203bd7728540) by [Tianlei Wu](https://medium.com/@tianlei.wu).


# ## Basic setup
from __future__ import annotations

import io
import os
import time
from pathlib import Path

from modal import Image, Secret, Stub, method

# Create a Stub representing a Modal app.

stub = Stub("stable-diffusion-onnx")

# ## Model dependencies
#
# We will install diffusers and the ONNX runtime GPU dependencies.

model_id = "tlwu/stable-diffusion-v1-5"
cache_path = "/vol/cache"


def download_models():
    import diffusers

    hugging_face_token = os.environ["HUGGINGFACE_TOKEN"]

    # Download models from the HunggingFace Hub and store
    # in local image cache.
    pipe = diffusers.OnnxStableDiffusionPipeline.from_pretrained(
        model_id,
        revision="fp16",
        provider="CUDAExecutionProvider",
        use_auth_token=hugging_face_token,
    )
    pipe.save_pretrained(cache_path, safe_serialization=True)


image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "diffusers[torch]>=0.15.1",
        "transformers==4.26.0",
        "safetensors",
        "torch>=2.0",
        "onnxruntime-gpu>=1.14",
    )
    .run_function(
        download_models,
        secrets=[Secret.from_name("huggingface-secret")],
        gpu="A10G",
    )
)
stub.image = image

# ## Load model and run inference
#
# We'll use the [container lifecycle `__enter__` method](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta) to load the model
# pipeline and then run inference using the `run_inference` method.


@stub.cls(gpu="A10G")
class StableDiffusion:
    def __enter__(self):
        import diffusers

        self.pipe = diffusers.OnnxStableDiffusionPipeline.from_pretrained(
            cache_path,
            revision="fp16",
            provider="CUDAExecutionProvider",
            device_map="auto",
        )

    @method()
    def run_inference(
        self, prompt: str, steps: int = 20, batch_size: int = 4
    ) -> list[bytes]:
        # Run pipeline
        images = self.pipe(
            [prompt] * batch_size,
            num_inference_steps=steps,
            guidance_scale=7.0,
        ).images

        # Convert to PNG bytes
        image_output = []
        for image in images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                image_output.append(buf.getvalue())
        return image_output


# Call this script with the Modal CLI: `modal run stable_diffusion_cli.py --prompt "a photo of a castle floating on clouds"`


@stub.local_entrypoint()
def entrypoint(
    prompt: str = "martha stewart at burning man",
    samples: int = 5,
    steps: int = 10,
    batch_size: int = 1,
):
    print(
        f"prompt => {prompt}, steps => {steps}, samples => {samples}, batch_size => {batch_size}"
    )

    dir = Path("/tmp/stable-diffusion")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    sd = StableDiffusion()
    for i in range(samples):
        t0 = time.time()
        images = sd.run_inference.call(prompt, steps, batch_size)
        total_time = time.time() - t0
        print(
            f"Sample {i} took {total_time:.3f}s ({(total_time)/len(images):.3f}s / image)."
        )
        for j, image_bytes in enumerate(images):
            output_path = dir / f"output_{j}_{i}.png"
            print(f"Saving it to {output_path}")
            with open(output_path, "wb") as f:
                f.write(image_bytes)
