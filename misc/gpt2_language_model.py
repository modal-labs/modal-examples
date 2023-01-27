# ---
# args: ["--prompt", "test prompt for symon"]
# ---
import modal

stub = modal.Stub("example-gpt2")

volume = modal.SharedVolume().persist("gpt2")

CACHE_PATH = "/root/model_cache"


@stub.function(
    image=modal.Image.debian_slim().pip_install("torch", "transformers"),
    shared_volumes={CACHE_PATH: volume},
    secret=modal.Secret({"TRANSFORMERS_CACHE": CACHE_PATH}),
)
def generate_text(prompt: str):
    from transformers import pipeline

    generator = pipeline("text-generation", model="gpt2")
    return generator(prompt, do_sample=True, min_length=50, max_length=250)[0][
        "generated_text"
    ]
