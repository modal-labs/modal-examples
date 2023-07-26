# ---
# args: ["--prompt", "test prompt for symon"]
# ---
import modal

stub = modal.Stub("example-gpt2")

CACHE_PATH = "/root/model_cache"


# Run as a build function to save the model files into the custom `modal.Image`.
def download_model():
    from transformers import pipeline

    generator = pipeline("text-generation", model="gpt2")
    generator.save_pretrained(CACHE_PATH)


@stub.function(
    image=modal.Image.debian_slim()
    .pip_install("torch", "transformers")
    .run_function(download_model),
)
def generate_text(prompt: str):
    from transformers import pipeline

    # NOTE: This model load runs on every function invocation. It's more efficient
    # to use lifecycle methods: modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta.
    generator = pipeline("text-generation", model=CACHE_PATH)
    return generator(prompt, do_sample=True, min_length=50, max_length=250)[0][
        "generated_text"
    ]


@stub.local_entrypoint()
def main(prompt: str = ""):
    generation = generate_text.call(prompt=prompt or "Show me the meaning of")
    print(generation)
