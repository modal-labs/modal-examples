# ---
# deploy: true
# ---

# # Low Latency LLM inference with TensorRT-LLM (LLaMA 3 8B)

import os
import time
from pathlib import Path

import modal

here = Path(__file__).parent

MINUTES = 60  # seconds

N_GPUS = 1  # Heads up: this example has not yet been tested with multiple GPUs
GPU_CONFIG = f"H100:{N_GPUS}"
ALLOW_CONCURRENT_INPUTS = 1

app_name = "example-trtllm-inference"
app = modal.App(app_name)

volume = modal.Volume.from_name(
    f"{app_name}-volume", create_if_missing=True
)
VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"
MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"  # fork without repo gating

flash_attn_release = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp313-cp313-linux_x86_64.whl"
)
tensorrt_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",  # TRT-LLM requires Python 3.10 or 3.12
    ).entrypoint(
        [] # remove verbose logging by base image on entry
    ).apt_install(
        "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget"
    ).pip_install(
        "tensorrt-llm==0.18.0.dev2025031100",
        "pynvml<12",  # avoid breaking change to pynvml version API
        pre=True,
        extra_index_url="https://pypi.nvidia.com",
    ).pip_install(
        "hf-transfer==0.1.9",
        "huggingface_hub==0.28.1",
    ).pip_install(
        "flashinfer-python",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.4/"
    ).env(
        {
         "HF_HUB_ENABLE_HF_TRANSFER": "1",
         "HF_HOME": str(MODELS_PATH),
        }
    )
)

def get_quant_config():
    from tensorrt_llm.llmapi import QuantConfig
    return QuantConfig(quant_algo="FP8")

def get_calib_config():
    from tensorrt_llm.llmapi import CalibConfig
    return CalibConfig(
        calib_batches=512,
        calib_batch_size=1,
        calib_max_seq_length=2048,
        tokenizer_max_seq_length=4096
    )

def get_plugin_config():
    from tensorrt_llm.plugin.plugin import PluginConfig
    return PluginConfig.from_dict({
        "multiple_profiles": True,
        "paged_kv_cache": True,
        "use_paged_context_fmha": True,
        "low_latency_gemm_swiglu_plugin": "fp8",
        "low_latency_gemm_plugin": "fp8"
    })

def get_build_config():
    from tensorrt_llm import BuildConfig
    return BuildConfig(
        plugin_config=get_plugin_config(),
        speculative_decoding_mode="LOOKAHEAD_DECODING",
        max_input_len=32768,
        max_num_tokens=65536,
        max_batch_size=ALLOW_CONCURRENT_INPUTS,
    )

def get_speculative_config():
    from tensorrt_llm.llmapi import LookaheadDecodingConfig
    return LookaheadDecodingConfig(
        max_window_size=8,
        max_ngram_size=6,
        max_verification_set_size=8,
    )

with tensorrt_image.imports():
    import torch
    from tensorrt_llm import LLM, SamplingParams

@app.cls(
    image=tensorrt_image,
    allow_concurrent_inputs=ALLOW_CONCURRENT_INPUTS,
    gpu=GPU_CONFIG,
    scaledown_window=10 * MINUTES,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class Model:
    def build_engine(self, engine_path, engine_kwargs) -> None:
        llm = LLM(model=self.model_path, **engine_kwargs)
        llm.save(engine_path)
        return llm

    @modal.enter()
    def enter(self):
        from transformers import AutoTokenizer
        from huggingface_hub import snapshot_download

        self.model_path = MODELS_PATH / MODEL_ID

        print("downloading base model if necessary")
        snapshot_download(MODEL_ID, local_dir=self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        engine_kwargs = {
            "quant_config": get_quant_config(),
            "calib_config": get_calib_config(),
            "build_config": get_build_config(),
            "speculative_config": get_speculative_config(),
            "tensor_parallel_size": torch.cuda.device_count(),
        }

        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024, # max generated tokens
            lookahead_config=engine_kwargs["speculative_config"]
        )

        engine_path = self.model_path / "trtllm_engine"
        if not os.path.exists(engine_path):
            print(f"building new engine at {engine_path}")
            self.llm = self.build_engine(engine_path, engine_kwargs)
        else:
            print (f"loading engine from {engine_path}")
            self.llm = LLM(model=engine_path, **engine_kwargs)

    @modal.method()
    def generate(self, prompt) -> dict:
        messages = [
            {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        output = self.llm.generate(text, self.sampling_params)
        return output.outputs[0].text

    @modal.method()
    def noop(self):
        pass

    @modal.exit()
    def shutdown(self):
        self.llm.shutdown()
        del self.llm


@app.local_entrypoint()
def main():
    prompts = [
        "What is the meaning of life?",
        "What are the differences between Javascript and Python?",
        "Create a short story about a society where people can only speak in metaphors.",
        "Discuss the effects of the Industrial Revolution on urban development in 19th century Europe.",
        "What is the product of 9 and 8?",
    ]

    model = Model()

    # Cold boot the container
    model.noop.remote()

    print_queue = []
    for prompt in prompts:
        start_time = time.perf_counter()
        generated_text = model.generate.remote(prompt)
        latency_ms = (time.perf_counter() - start_time) * 1000

        print_queue.append((prompt, generated_text, latency_ms))

    time.sleep(3)
    for prompt, generated_text, latency_ms in print_queue:
        print(f"Processed prompt in {latency_ms:.2f}ms")
        print(f"Prompt: {prompt}")
        print(f"Generated Text: {generated_text}")
        print("-" * 80)
