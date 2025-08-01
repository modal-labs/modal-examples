# ---
# args: ["--prompt", "How do planes work?"]
# ---
# # Run Falcon-40B with bitsandbytes
#
# In this example, we download the full-precision weights of the Falcon-40B LLM but load it in 4-bit using
# Tim Dettmers' [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes) library. This enables it to fit
# into a single GPU (A100 40GB).
#
# Due to the current limitations of the library, the inference speed is a little over 2 tokens/second and due
# to the sheer size of the model, the cold start time on Modal is around 2 minutes.
#
# For faster cold start at the expense of inference speed, check out
# [Running Falcon-40B with AutoGPTQ](https://modal.com/docs/examples/falcon_gptq).
#
# ## Setup
#
# First we import the components we need from `modal`.

from typing import Optional

import modal


# Spec for an image where falcon-40b-instruct is cached locally
def download_falcon_40b():
    from huggingface_hub import snapshot_download

    model_name = "tiiuae/falcon-40b-instruct"
    snapshot_download(model_name)


image = (
    modal.Image.micromamba()
    .micromamba_install(
        "cudatoolkit=11.7",
        "cudnn=8.1.0",
        "cuda-nvcc",
        "scipy",
        channels=["conda-forge", "nvidia"],
    )
    .apt_install("git")
    .pip_install(
        "bitsandbytes==0.39.0",
        "bitsandbytes-cuda117==0.26.0.post2",
        "peft==0.6.2",
        "transformers==4.31.0",
        "accelerate==0.26.1",
        "hf-transfer==0.1.5",
        "torch==2.0.0",
        "torchvision==0.15.1",
        "sentencepiece==0.1.97",
        "huggingface_hub==0.14.1",
        "einops==0.6.1",
    )
    # Use huggingface's hi-perf hf-transfer library to download this large model.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_falcon_40b)
)

app = modal.App(image=image, name="example-falcon-bnb")


# ## The model class
#
# Next, we write the model code. We want Modal to load the model into memory just once every time a container starts up,
# so we use [class syntax](https://modal.com/docs/guide/lifecycle-functions) and the `@enter` decorator.
#
# Within the [@app.cls](https://modal.com/docs/reference/modal.App#cls) decorator, we use the [gpu parameter](/docs/guide/gpu)
# to specify that we want to run our function on an [A100 GPU](https://modal.com/docs/guide/gpu). We also allow each call 10 mintues to complete,
# and request the runner to stay live for 5 minutes after its last request.
#
# We load the model in 4-bit using the `bitsandbytes` library.
#
# The rest is just using the [`pipeline`](https://huggingface.co/docs/transformers/en/main_classes/pipelines)
# abstraction from the `transformers` library. Refer to the documentation for more parameters and tuning.
@app.cls(
    gpu="A100",
    timeout=60 * 10,  # 10 minute timeout on inputs
    scaledown_window=60 * 5,  # Keep runner alive for 5 minutes
)
class Falcon40B_4bit:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        model_name = "tiiuae/falcon-40b-instruct"

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,  # Model is downloaded to cache dir
            device_map="auto",
            quantization_config=nf4_config,
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",
        )
        tokenizer.bos_token_id = 1

        self.model = torch.compile(model)
        self.tokenizer = tokenizer

    @modal.method()
    def generate(self, prompt: str):
        from threading import Thread

        from transformers import GenerationConfig, TextIteratorStreamer

        tokenized = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized.input_ids
        input_ids = input_ids.to(self.model.device)

        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.1,
            max_new_tokens=512,
        )

        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        generate_kwargs = dict(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            attention_mask=tokenized.attention_mask,
            output_scores=True,
            streamer=streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()
        for new_text in streamer:
            print(new_text, end="")
            yield new_text

        thread.join()


# ## Run the model
# We define a [`local_entrypoint`](https:modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run -q falcon_bitsandbytes.py`. The `-q` flag
# enables streaming to work in the terminal output.
prompt_template = (
    "A chat between a curious human user and an artificial intelligence assistant. The assistant give a helpful, detailed, and accurate answer to the user's question."
    "\n\nUser:\n{}\n\nAssistant:\n"
)


@app.local_entrypoint()
def cli(prompt: Optional[str] = None):
    question = (
        prompt
        or "What are the main differences between Python and JavaScript programming languages?"
    )
    model = Falcon40B_4bit()
    for text in model.generate.remote_gen(prompt_template.format(question)):
        print(text, end="", flush=True)


# ## Serve the model
# Finally, we can serve the model from a web endpoint with `modal deploy falcon_bitsandbytes.py`. If
# you visit the resulting URL with a question parameter in your URL, you can view the model's
# stream back a response.
# You can try our deployment [here](https://modal-labs--example-falcon-bnb-get.modal.run/?question=How%20do%20planes%20work?).
@app.function(timeout=60 * 10)
@modal.fastapi_endpoint()
def get(question: str):
    from itertools import chain

    from fastapi.responses import StreamingResponse

    model = Falcon40B_4bit()
    return StreamingResponse(
        chain(
            ("Loading model (100GB). This usually takes around 110s ...\n\n"),
            model.generate.remote(prompt_template.format(question)),
        ),
        media_type="text/event-stream",
    )
