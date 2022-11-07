import modal
import sys

stub = modal.Stub("example-gpt2")

volume = modal.SharedVolume().persist("gpt2")

CACHE_PATH = "/root/model_cache"


@stub.function(
    image=modal.Image.debian_slim().pip_install(["torch", "transformers"]),
    shared_volumes={CACHE_PATH: volume},
    secret=modal.Secret({"TRANSFORMERS_CACHE": CACHE_PATH}),
)
def generate_text(prompt):
    from transformers import pipeline

    generator = pipeline("text-generation", model="gpt2")
    return generator(prompt, do_sample=True, min_length=50)[0]["generated_text"]


if __name__ == "__main__":
    with stub.run():
        print(generate_text(sys.argv[1]))
