# ---
# integration-test: false
# ---
# # Run Falcon-40B in one GPU with AutoGPTQ

# In this example, we run a quantized version of Falcon-40B, the first open-source large language
# model of its size, using HuggingFace's [transformers](https://huggingface.co/docs/transformers/index)
# library and AutoGPTQ.
#
# ## Setup
#
# First we import the components we need from `modal`.

from modal import Image, gpu, Stub, method

# ## Define a container image
#
# To take advantage of Modal's blazing fast cold-start times, we download model weights
# into a folder inside our container image. These weights come from a quantized model
# found on Huggingface.
IMAGE_MODEL_DIR = "/model"
def download_model():
    from huggingface_hub import snapshot_download

    model_name = "TheBloke/falcon-40b-instruct-GPTQ"
    snapshot_download(model_name, local_dir=IMAGE_MODEL_DIR)

# Now, we define our image. We'll use the `debian-slim` base image, and install the dependencies we need
# using [`pip_install`](/docs/reference/modal.Image#pip_install). At the end, we'll use
# [`run_function`](/docs/guide/custom-container#running-a-function-as-a-build-step-beta) to run the
# function defined above as part of the image build.

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "huggingface_hub==0.14.1",
        "transformers @ git+https://github.com/huggingface/transformers.git",
        "auto-gptq @ git+https://github.com/PanQiWei/AutoGPTQ.git",
        "einops==0.6.1"
    )
    .run_function(download_model)
)

# Let's instantiate and name our [Stub](/docs/guide/apps).
stub = Stub(image=image)

# ## The model class
#
# Next, we write the model code. We want Modal to load the model into memory just once every time a container starts up,
# so we use [class syntax](/docs/guide/lifecycle-functions) and the `__enter__` method.
#
# Within the [@stub.cls](/docs/reference/modal.Stub#cls) decorator, we use the [gpu parameter](/docs/guide/gpu)
# to specify that we want to run our function on an [A100 GPU](/pricing).
#
# The rest is just using the [pipeline()](https://huggingface.co/docs/transformers/en/main_classes/pipelines)
# abstraction from the `transformers` library. Refer to the documentation for more parameters and tuning.
@stub.cls(gpu=gpu.A100())
class Falcon40BGPTQ:
    def __enter__(self):
        from transformers import AutoTokenizer, pipeline
        from auto_gptq import AutoGPTQForCausalLM

        model_basename = "gptq_model-4bit--1g"

        use_triton = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            IMAGE_MODEL_DIR,
            use_fast=True
        )

        self.model = AutoGPTQForCausalLM.from_quantized(
            IMAGE_MODEL_DIR,
            trust_remote_code=True,
            use_safetensors=True,
            model_basename=model_basename,
            device_map="auto",
            use_triton=use_triton,
            strict=False,
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.5,
            repetition_penalty=1.15
        )

        print("Loaded model.")

    @method()
    def generate(self, prompt: str):
        print(self.pipe(prompt)[0]['generated_text'])

# ## Run the model
# Finally, we define a [`local_entrypoint`](/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run falcon-gptq.py`.
@stub.local_entrypoint()
def test():
    prompt = """A chat between a curious human user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    User: What are the main differences between Python and JavaScript programming languages?
    Assistant: """

    model = Falcon40BGPTQ()
    model.generate.call(prompt)
