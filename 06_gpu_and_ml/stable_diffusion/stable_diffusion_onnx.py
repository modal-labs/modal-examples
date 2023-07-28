# # Stable Diffusion inference using the ONNX Runtime
#
# This example is similar to the example [Stable Diffusion CLI](/docs/guide/ex/stable_diffusion_cli)
# but running inference unsing the [ONNX Runtime](https://onnxruntime.ai/) instead of PyTorch.
#
# Inference with ONNX is faster by about ~100ms, with throughput of ~950ms / image on a
# A10G GPU. Cold boot times are higher, however; taking about ~18s for the models to
# be loaded into memory.


# ## Basic setup

import io
import time
from pathlib import Path

from modal import Image, Stub, method

# Create a Stub representing a Modal app.

stub = Stub("stable-diffusion-onnx")

# ## Model dependencies
#
# We will install `optimum` and the ONNX runtime GPU package.

model_id = "runwayml/stable-diffusion-v1-5"
cache_path = "/vol/cache"


def download_models():
    from optimum.onnxruntime import ORTStableDiffusionPipeline

    pipe = ORTStableDiffusionPipeline.from_pretrained(
        model_id, revision="fp16", export=True
    )
    pipe.save_pretrained(cache_path, safe_serialization=True)


stub.image = (
    Image.debian_slim(python_version="3.11")
    .pip_install("diffusers~=0.19.1", "optimum[onnxruntime-gpu]~=1.10.1")
    .run_function(download_models)
)

# ## Load model and run inference
#
# The container lifecycle [`__enter__` function](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta)
# loads the model at startup. Then, we evaluate it in the `run_inference` function.


@stub.cls(gpu="A10G")
class StableDiffusion:
    def __enter__(self):
        from optimum.onnxruntime import ORTStableDiffusionPipeline

        self.pipe = ORTStableDiffusionPipeline.from_pretrained(
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


# Call this script with the Modal CLI: `modal run stable_diffusion_cli.py --prompt "a photo of a castle floating on clouds"`.


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
