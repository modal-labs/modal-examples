# ---
# integration-test: false
# ---
# # Run Bloom Chat 176B model (ChatGPT-size) with AutoGPTQ

# In this example, we run a quantized 4-bit version of Sambanova Systems's BLOOMChat 1.0, the only open-source large language
# model of the size matching GPT 3.5 model used in the well-known ChatGPT, using HuggingFace's [transformers](https://huggingface.co/docs/transformers/index) library and [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ).
# 
# We use the quantized version of the model created by TheBloke [BLOOMChat-176B-v1-GPTQ](https://huggingface.co/TheBloke/BLOOMChat-176B-v1-GPTQ). 
# It is a great idea to checkout his Model Card (`README.md`) for more details on the model. 
# It is also possible to run the original BLOOM model as well [bloomz-176B-GPTQ](https://huggingface.co/TheBloke/bloomz-176B-GPTQ).
#
# Due its enormous size the model files are around 100GBs and they take a while to download and set up. Occasionally there may be a network error. 
# Please kindly retry the process. The build shouldn't take more than 20 minutes. 
# This example includes verbose feedback to help you track the progress. 
# 
# Cold boot time on 5x A100 40GB GPUs is around 90s.
#
# ## Setup
#
# First we import the components we need from `modal`. We also import `time` utility to provide us with insights into the performance.

import time
from modal import Image, method, Stub, web_endpoint, gpu

stub = Stub(name="example-bloom-gptq") # change this to your own name

IMAGE_MODEL_DIR = "/model" 
MODEL_BASE_FILE = "gptq_model-4bit--1g" # the model file name without the ".safetensors" suffix

# Bloom models were split to pass Hugging Face's 50GB limit. Therefore after downloading we will merge the model files into a single file. 
SPLIT_FILE_REGEX = "gptq_model-4bit--1g.JOINBEFOREUSE.split-*.safetensors"
command = f"cd {IMAGE_MODEL_DIR} && cat {SPLIT_FILE_REGEX} > {MODEL_BASE_FILE}.safetensors && rm {SPLIT_FILE_REGEX}"


# Here we declare a function to download the model during the build time.
def download_model():
    import transformers
    from huggingface_hub import snapshot_download
    MODEL_NAME = "TheBloke/BLOOMChat-176B-v1-GPTQ"

    # Verify at least 200GB is available.
    # This is to ensure that the model can be downloaded and merged.
    # The merged file will be deleted after the model is loaded.
    import shutil
    total, used, free = shutil.disk_usage("/")
    assert free > 200 * 1024 * 1024 * 1024, f"Expected at least 200GB free space. Got {free}"

    # The download may fail once in a while. Simply rerun the script again.
    print(f"Downloading model... expect 3-15 minutes...")
    start_time = time.time()
    snapshot_download(MODEL_NAME, 
        local_dir=IMAGE_MODEL_DIR,
        resume_download=True,
        # token is optional but it will speed up the download
        # token="hf_xxx"
    )
    end_time = time.time()
    print(f"Download completed, took => {end_time - start_time:.2f}s")

    print("Combining model files... expect 3 minutes...")
    import subprocess
    subprocess.run(command, check=True, shell=True)
    print(f"Model files combined, took => {time.time() - end_time:.2f}s")
    end_time = time.time()

    # We move cache to avoid doing that during inference time.
    print("Moving cache... expect 2-4 minutes...")
    transformers.utils.move_cache()
    print(f"Cache moved, took => {time.time() - end_time:.2f}s")

    print("Done! Modal may take up to 15 minutes to upload a snapshot...")


inference_image = (
    Image.from_dockerhub(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04",
        setup_dockerfile_commands=[
            "RUN apt-get update",
            "RUN apt-get install -y python3 python3-pip python-is-python3",
        ],
    )
    .apt_install("git", "gcc", "build-essential")
    .run_commands(
        "pip install --compile huggingface_hub transformers torch einops hf_transfer",
    )
    .env({
            "HF_HUB_ENABLE_HF_TRANSFER": "1", # enable fast downloads, this mediates common Hugging Face Read Timeouts 
            "PIP_NO_CACHE_DIR": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "SAFETENSORS_FAST_GPU": "1", # Load the model directly to GPU memory skipping RAM
            "BITSANDBYTES_NOWELCOME": "1",
        })
    .run_function(download_model)
    .run_commands(
        # It appears that installing directly through pip torch extension fails to compile. 
        # As such, we clone it and install it from source whilst providing T4 GPU.
        "git clone https://github.com/PanQiWei/AutoGPTQ.git",
        "cd AutoGPTQ && pip install --compile .",
        gpu="T4",
    )
)

api_image = (
    Image.debian_slim()
)


@stub.cls(image=inference_image, gpu=gpu.A100(count=5), container_idle_timeout=300, cloud="oci", concurrency_limit=1)
class BloomChat:
    def __enter__(self):
        start_import = time.time()
        import torch
        from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
        from auto_gptq import AutoGPTQForCausalLM

        print(f"importing libraries took => {time.time() - start_import:.2f}s")

        start_load_tokenizer = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            IMAGE_MODEL_DIR, use_fast=True
        )
        print(f"loading tokenizer took => {time.time() - start_load_tokenizer:.2f}s")

        start_loading_model = time.time()
        print("loading model...")

        self.model = AutoGPTQForCausalLM.from_quantized(
            IMAGE_MODEL_DIR,
            model_basename=MODEL_BASE_FILE,
            use_safetensors=True,
            device_map="auto",
            use_triton=False,
            strict=True,
        )
        self.model.tie_weights()

        print(f"Model loaded in =>  {time.time() - start_loading_model:.2f}s")
        
        cold_boot_time = time.time() - start_import
        print(f"total cold boot time  => {cold_boot_time:.2f}s")

        self.is_loaded = False
        self.cold_boot_time = cold_boot_time

    @method()
    async def generate(self, input, temperature = 0.7, max_tokens = 256, stop_words = [""]):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        # Cold boot time, should be zero if model is already loaded
        if (self.is_loaded == False):
            cold_boot_time = self.cold_boot_time
            self.is_loaded = True
        else:
            cold_boot_time = 0

        stop_token_ids = self.tokenizer.convert_tokens_to_ids(stop_words)
        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                for stop_id in stop_token_ids:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False
        stopping_criteria = StoppingCriteriaList([StopOnTokens()])

        t3 = time.time()
        input_ids = self.tokenizer(input, return_tensors='pt').input_ids.cuda()
        input_tokens= len(input_ids[0])
        generation = self.model.generate(inputs=input_ids, 
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            max_new_tokens=max_tokens,
            repetition_penalty=1.1,
            stopping_criteria=stopping_criteria if len(stop_words) > 0 else None,
        )
        completion_tokens = len(generation[0]) - input_tokens

        # Provide completion without the prompt
        # Subtract the input tokens from the generated tokens
        new_tokens = generation[0][input_tokens:]

        completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        latency = time.time() - t3

        print(f"Input tokens: {input_tokens}")
        print(f"Completion tokens: {completion_tokens}")
        print(f"Generation took => {latency:.2f}s")
        
        return {
            "completion": completion,
            "completion_tokens": completion_tokens,
            "prompt_tokens": input_tokens,
            "execution_time": latency,
            "delay_time": cold_boot_time,
            "model": stub.name,
        }



DEMO_INPUT = """
<human>: What is Modal?
<bot>: Modal (modal.com) lets you run code in the cloud without having to think about infrastructure.
Features
- Run any code remotely within seconds.
- Define container environments in code (or use one of our pre-built backends).
- Scale up horizontally to thousands of containers.
- Deploy and monitor persistent cron jobs.
- Attach GPUs with a single line of code.
- Serve your functions as web endpoints.
- Use powerful primitives like distributed dictionaries and queues.
- Run your code on a schedule.
<human>: What is the future of Modal?
<bot>: 
"""

@stub.local_entrypoint()
def main():
    t0 = time.time()
    model = BloomChat()
    val= model.generate.call(DEMO_INPUT)
    print(val)
    print(f"Total time: {time.time() - t0:.2f}s")


from pydantic import BaseModel
from typing_extensions import Annotated
from typing import List, Union

class CompletionRequest(BaseModel):
    prompt: Annotated[str, "The prompt for text completion"]
    temperature: Annotated[
        float,
        "Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic.",
    ] = 0.7
    max_tokens: Annotated[
        int, "Maximum number of new tokens to generate for text completion."
    ] = 16
    stop_words: Annotated[Union[str, List[str]], "Any additional stop words."] = []
    ref: Annotated[str, "Reference string for the completion"] = ""


@stub.function(image=api_image, cloud="oci", concurrency_limit=1)
@web_endpoint(method="POST")
def api(request: CompletionRequest):
    t = time.time()
    print(f"Request received: {request.ref}")
    result = BloomChat().generate.call(input=request.prompt, 
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop_words=request.stop_words
            )

    result["ref"] = request.ref

    print(f"Request completed: {request.ref} => {time.time() - t:.2f}s")

    return result