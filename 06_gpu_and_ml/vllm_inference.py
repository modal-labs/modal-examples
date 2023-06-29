# ---
# integration-test: false
# ---
# # Run inference with vLLM

# In this example, we show how to run basic inference, using `[vLLM](https://github.com/vllm-project/vllm)`
# to take advantage of PagedAttention, which speeds up sequential inferences with optimized key-value caching.
#
# `vLLM` also supports a use case as a FastAPI server which we will explore in a future guide. This example
# walks through setting up an environment that works with `vLLM ` for basic inference.
#
# One can expect 32 second cold starts and 120 tokens/second during inference. The example generates around 1500 tokens in 12 seconds.
#
# ## Setup
#
# First we import the components we need from `modal`.

from modal import Stub, Image, method


# We now want to create a Modal image which has the model weights pre-saved. The benefit of this
# is that the container no longer has to re-download the model from Huggingface - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# Since `vLLM` uses the default Huggingface cache location, so we can use library functions to pre-download the model into our image.
def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download("lmsys/vicuna-13b-v1.3")
    snapshot_download("hf-internal-testing/llama-tokenizer")


image = (
    Image.from_dockerhub("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "vllm @ git+https://github.com/vllm-project/vllm.git@2b7d3aca2e1dd25fe26424f57c051af3b823cd71"
    )
    .run_function(download_model)
)

stub = Stub(image=image)


# ## The model class
#
# Next, we write the model code. We want Modal to load the model into memory just once every time a container starts up,
# so we use [class syntax](/docs/guide/lifecycle-functions) and the __enter__` method.
#
# The `vLLM` allows the inference code to remain quite clean.
@stub.cls(gpu="A100")
class Model:
    def __enter__(self):
        from vllm import LLM

        self.llm = LLM(model="lmsys/vicuna-13b-v1.3")  # Load the model
        self.template = (
            "You are a helpful assistant.\n\n### USER:\n{}\n### ASSISTANT:\n"
        )

    @method()
    def generate(self, user_questions):
        from vllm import SamplingParams

        prompts = [self.template.format(q) for q in user_questions]
        sampling_params = SamplingParams(
            temperature=0.8, top_p=0.95, max_tokens=512
        )
        result = self.llm.generate(
            prompts, sampling_params
        )  # Trigger inference
        for output in result:
            print(output.prompt, output.outputs[0].text, "\n\n", sep="")


# ## Run the model
# We define a [`local_entrypoint`](/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run vllm_inference.py`.
@stub.local_entrypoint()
def main():
    model = Model()
    model.generate.call(
        [
            "Implement a Python function to compute the Fibonacci numbers.",
            "Write a Rust function that performs fast exponentiation.",
            "What is the fable involving a fox and grapes?",
        ]
    )
