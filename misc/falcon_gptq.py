# # Run Falcon-40B with AutoGPTQ
#
# In this example, we run a quantized 4-bit version of Falcon-40B, the first open-source large language
# model of its size, using HuggingFace's [transformers](https://huggingface.co/docs/transformers/index)
# library and [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ).
#
# Due to the current limitations of the library, the inference speed is a little under 1 token/second and the
# cold start time on Modal is around 25s.
#
# For faster inference at the expense of a slower cold start, check out
# [Running Falcon-40B with `bitsandbytes` quantization](https://modal.com/docs/examples/falcon_bitsandbytes). You can also
# run a smaller model via the [Gemma 7B example](https://modal.com/docs/examples/vllm_gemma).
#
# ## Setup
#
# First we import the components we need from `modal`.

import modal

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
# using [`pip_install`](https://modal.com/docs/reference/modal.Image#pip_install). At the end, we'll use
# [`run_function`](https://modal.com/docs/guide/custom-container#run-a-modal-function-during-your-build-with-run_function-beta) to run the
# function defined above as part of the image build.

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "auto-gptq==0.7.0",
        "einops==0.6.1",
        "hf-transfer==0.1.5",
        "huggingface_hub==0.14.1",
        "transformers==4.31.0",
    )
    # Use huggingface's hi-perf hf-transfer library to download this large model.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model)
)

# Let's instantiate and name our [`App`](https://modal.com/docs/guide/apps).
app = modal.App(name="example-falcon-gptq", image=image)


# ## The model class
#
# Next, we write the model code. We want Modal to load the model into memory just once every time a container starts up,
# so we use [class syntax](https://modal.com/docs/guide/lifecycle-functions) and the `@enter` decorator.
#
# Within the [`@app.cls`](https://modal.com/docs/reference/modal.App#cls) decorator, we use the [`gpu` parameter](https://modal.com/docs/guide/gpu)
# to specify that we want to run our function on an [A100 GPU](https://modal.com/docs/guide/gpu#a100-gpus). We also allow each call 10 mintues to complete,
# and request the runner to stay live for 5 minutes after its last request.
#
# The rest is just using the `transformers` library to run the model. Refer to the
# [documentation](https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationMixin.generate)
# for more parameters and tuning.
#
# Note that we need to create a separate thread to call the `generate` function because we need to
# yield the text back from the streamer in the main thread. This is an idiosyncrasy with streaming in `transformers`.
@app.cls(gpu="A100", timeout=60 * 10, scaledown_window=60 * 5)
class Falcon40BGPTQ:
    @modal.enter()
    def load_model(self):
        from auto_gptq import AutoGPTQForCausalLM
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(IMAGE_MODEL_DIR, use_fast=True)
        print("Loaded tokenizer.")

        self.model = AutoGPTQForCausalLM.from_quantized(
            IMAGE_MODEL_DIR,
            trust_remote_code=True,
            use_safetensors=True,
            device_map="auto",
            use_triton=False,
            strict=False,
        )
        print("Loaded model.")

    @modal.method()
    def generate(self, prompt: str):
        from threading import Thread

        from transformers import TextIteratorStreamer

        inputs = self.tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        generation_kwargs = dict(
            inputs=inputs.input_ids.cuda(),
            attention_mask=inputs.attention_mask,
            temperature=0.1,
            max_new_tokens=512,
            streamer=streamer,
        )

        # Run generation on separate thread to enable response streaming.
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            yield new_text

        thread.join()


# ## Run the model
# We define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run -q falcon_gptq.py`. The `-q` flag
# enables streaming to work in the terminal output.
prompt_template = (
    "A chat between a curious human user and an artificial intelligence assistant. The assistant give a helpful, detailed, and accurate answer to the user's question."
    "\n\nUser:\n{}\n\nAssistant:\n"
)


@app.local_entrypoint()
def cli():
    question = "What are the main differences between Python and JavaScript programming languages?"
    model = Falcon40BGPTQ()
    for text in model.generate.remote_gen(prompt_template.format(question)):
        print(text, end="", flush=True)


# ## Serve the model
# Finally, we can serve the model from a web endpoint with `modal deploy falcon_gptq.py`. If
# you visit the resulting URL with a question parameter in your URL, you can view the model's
# stream back a response.
# You can try our deployment [here](https://modal-labs--example-falcon-gptq-get.modal.run/?question=Why%20are%20manhole%20covers%20round?).
@app.function(timeout=60 * 10)
@modal.fastapi_endpoint()
def get(question: str):
    from itertools import chain

    from fastapi.responses import StreamingResponse

    model = Falcon40BGPTQ()
    return StreamingResponse(
        chain(
            ("Loading model. This usually takes around 20s ...\n\n"),
            model.generate.remote_gen(prompt_template.format(question)),
        ),
        media_type="text/event-stream",
    )
