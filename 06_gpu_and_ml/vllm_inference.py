# ---
# integration-test: false
# ---
# # Run inference with vLLM

# In this example, we show how to run basic inference, using [`vLLM`](https://github.com/vllm-project/vllm)
# to take advantage of PagedAttention, which speeds up sequential inferences with optimized key-value caching.
#
# `vLLM` also supports a use case as a FastAPI server which we will explore in a future guide. This example
# walks through setting up an environment that works with `vLLM ` for basic inference.
#
# One can expect 30 second cold starts and 120 tokens/second during inference. The example generates around 1000 tokens in 8 seconds.
#
# ## Setup
#
# First we import the components we need from `modal`.

from modal import Stub, Image, method


# ## Define a container image
#
# We want to create a Modal image which has the model weights pre-saved. The benefit of this
# is that the container no longer has to re-download the model from Huggingface - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# Since `vLLM` uses the default Huggingface cache location, we can use library functions to pre-download the model into our image.
def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download("lmsys/vicuna-13b-v1.3")
    snapshot_download("hf-internal-testing/llama-tokenizer")


# Now, we define our image. We’ll start from a Dockerhub image recommended by `vLLM`, upgrade the older
# version of `torch` to a new one specifically built for CUDA 11.8. Next, we install `vLLM` from source to get the latest updates.
# Finally, we’ll use run_function to run the function defined above to ensure the weights of the model
# are saved within the container image.
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
# The inference function is best represented with Modal's [class syntax](/docs/guide/lifecycle-functions) and the `__enter__` method.
# This enables us to load the model into memory just once every time a container starts up, and keep it cached
# on the GPU for each subsequent invocation of the function.
#
# The `vLLM` library allows the code to remain quite clean.
@stub.cls(gpu="A100")
class Model:
    def __enter__(self):
        from vllm import LLM

        self.llm = LLM(model="lmsys/vicuna-13b-v1.3")  # Load the model
        self.template = "You are a helpful assistant.\nUSER:\n{}\nASSISTANT:\n"

    @method()
    def generate(self, user_questions):
        from vllm import SamplingParams

        prompts = [self.template.format(q) for q in user_questions]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=800)
        result = self.llm.generate(prompts, sampling_params)
        for output in result:
            n += len(output.outputs[0].token_ids)
            print(output.prompt, output.outputs[0].text, "\n\n", sep="")


# ## Run the model
# We define a [`local_entrypoint`](/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run vllm_inference.py`.
@stub.local_entrypoint()
def main():
    model = Model()
    questions = [
        "Implement a Python function to compute the Fibonacci numbers.",
        "Write a Rust function that performs fast exponentiation.",
        "How do I allocate memory in C?",
        "What is the fable involving a fox and grapes?",
    ]
    model.generate.call(questions)
