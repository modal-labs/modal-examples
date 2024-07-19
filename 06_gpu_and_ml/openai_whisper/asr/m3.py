import modal

from typing import Annotated
from modal import Image

INFERENCE_PRECISION = "float16"
WEIGHT_ONLY_PRECISION = "int8"
MAX_BEAM_WIDTH = 4
MAX_BATCH_SIZE = 8
whisper_checkpoint_dir = f"whisper_large_v3_weights_{WEIGHT_ONLY_PRECISION}"
whisper_output_dir= f"whisper_large_v3_{WEIGHT_ONLY_PRECISION}"

LLAMA_MODEL_DIR = "/llama_8b_instruct_model"
LLAMA_ENGINE_DIR = "/llama_8b_instruct_engine"
LLAMA_MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
LLAMA_MODEL_REVISION = "b1532e4dee724d9ba63fe17496f298254d87ca64"  # pin model revisions to prevent unexpected changes!
LLAMA_CKPT_DIR = "/llama_8b_instruct_ckpt"
LLAMA_MAX_INPUT_LEN = 256
LLAMA_MAX_OUTPUT_LEN = 256
LLAMA_MAX_BATCH_SIZE = 128

DTYPE = "float16"

N_GPUS = 1
GPU_CONFIG = modal.gpu.A100(count=N_GPUS)

def download_llama():
    import os

    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(LLAMA_MODEL_DIR, exist_ok=True)
    snapshot_download(
        LLAMA_MODEL_ID,
        local_dir=LLAMA_MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
        revision=LLAMA_MODEL_REVISION,
    )
    move_cache()

def download_warmup_data():
    from datasets import load_dataset
    _ = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")


image = (
    Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10"
    )
    .apt_install(
      "openmpi-bin",
      "libopenmpi-dev",
      "git",
      "git-lfs",
      "wget",
    )
    .run_commands(  # get rid of CUDA banner
        "rm /opt/nvidia/entrypoint.d/10-banner.sh",
        "rm /opt/nvidia/entrypoint.d/12-banner.sh",
        "rm /opt/nvidia/entrypoint.d/15-container-copyright.txt",
        "rm /opt/nvidia/entrypoint.d/30-container-license.txt",
    )
    .run_commands(
        "wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken",
        "wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz",
        "wget --directory-prefix=assets https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav",
        "wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"
    )
    .pip_install_from_requirements("requirements.txt")
    .copy_local_file("convert_checkpoint.py")
    .run_commands(
        [
            f"python convert_checkpoint.py \
                --use_weight_only \
                --weight_only_precision {WEIGHT_ONLY_PRECISION} \
                --output_dir {whisper_checkpoint_dir}"
        ], gpu=GPU_CONFIG,
    )
    .run_commands(
        [
            f"trtllm-build --checkpoint_dir {whisper_checkpoint_dir}/encoder \
                  --output_dir {whisper_output_dir}/encoder \
                  --paged_kv_cache disable \
                  --moe_plugin disable \
                  --enable_xqa disable \
                  --use_custom_all_reduce disable \
                  --max_batch_size {MAX_BATCH_SIZE} \
                  --gemm_plugin disable \
                  --bert_attention_plugin {INFERENCE_PRECISION} \
                  --remove_input_padding disable \
                  --max_input_len 1500",
        ], gpu=GPU_CONFIG
    )
    .run_commands(
        [
            f"trtllm-build  --checkpoint_dir {whisper_checkpoint_dir}/decoder \
                  --output_dir {whisper_output_dir}/decoder \
                  --paged_kv_cache disable \
                  --moe_plugin disable \
                  --enable_xqa disable \
                  --use_custom_all_reduce disable \
                  --max_beam_width {MAX_BEAM_WIDTH} \
                  --max_batch_size {MAX_BATCH_SIZE} \
                  --max_seq_len 114 \
                  --max_input_len 14 \
                  --max_encoder_input_len 1500 \
                  --gemm_plugin {INFERENCE_PRECISION} \
                  --bert_attention_plugin {INFERENCE_PRECISION} \
                  --gpt_attention_plugin {INFERENCE_PRECISION} \
                  --remove_input_padding disable"
        ], gpu=GPU_CONFIG,
    )
    .copy_local_file("run.py", "/root/run.py")
    .copy_local_file("tokenizer.py", "/root/tokenizer.py")
    .copy_local_file("whisper_utils.py", "/root/whisper_utils.py")
    .pip_install("librosa")
    .pip_install_from_requirements("../llama/requirements.txt")
    .pip_install(
        "hf-transfer==0.1.6",
        "requests~=2.31.0",
    )
    .env(
        {"HF_HUB_ENABLE_HF_TRANSFER": "1"}  # enable hf-transfer for faster downloads
    )
    .run_function(
        download_llama,
        timeout=20 * 60,
    )
    .copy_local_file("../llama/convert_checkpoint.py", "/root/convert_checkpoint_llama.py")
    .run_commands(  # takes ~5 minutes
        [
            f"python /root/convert_checkpoint_llama.py \
                    --model_dir={LLAMA_MODEL_DIR} \
                    --output_dir={LLAMA_CKPT_DIR} \
                    --tp_size={N_GPUS} \
                    --dtype={DTYPE}",
        ],
        gpu=GPU_CONFIG,  # GPU must be present to load tensorrt_llm
    )
    .run_commands(  # takes ~5 minutes
        [
            f"trtllm-build \
                --checkpoint_dir {LLAMA_CKPT_DIR} \
                --output_dir {LLAMA_ENGINE_DIR} \
                --tp_size={N_GPUS} \
                --workers={N_GPUS} \
                --max_batch_size={LLAMA_MAX_BATCH_SIZE} \
                --max_input_len={LLAMA_MAX_INPUT_LEN} \
                --gemm_plugin={DTYPE} \
                --gpt_attention_plugin={DTYPE}",
        ],
        gpu=GPU_CONFIG,  # TRT-LLM compilation is GPU-specific, so make sure this matches production!
    ).env(  # show more log information from the inference engine
        {"TLLM_LOG_LEVEL": "INFO"}
    )
    .pip_install("pydantic==1.10.11")
    .run_function(
        download_warmup_data,
        timeout=20 * 60,
    )
)

COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}


def extract_assistant_response(output_text):
    """Model-specific code to extract model responses.

    See this doc for LLaMA 3: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/."""
    # Split the output text by the assistant header token
    parts = output_text.split("<|start_header_id|>assistant<|end_header_id|>")

    if len(parts) > 1:
        # Join the parts after the first occurrence of the assistant header token
        response = parts[1].split("<|eot_id|>")[0].strip()

        # Remove any remaining special tokens and whitespace
        response = response.replace("<|eot_id|>", "").strip()

        return response
    else:
        return output_text

app = modal.App("faster-v2", image=image)

@app.cls(keep_warm=1, allow_concurrent_inputs=1, concurrency_limit=2, gpu=GPU_CONFIG)
class Model:
    @modal.enter()
    def enter(self):
        self.assets_dir = "/assets"
        from run import WhisperTRTLLM, decode_dataset

        self.model = WhisperTRTLLM(f"/{whisper_output_dir}", assets_dir=self.assets_dir)

    @modal.enter()
    def load(self):
        import time

        print(
            f"{COLOR['HEADER']}🥶 Cold boot: spinning up TRT-LLM engine{COLOR['ENDC']}"
        )
        self.init_start = time.monotonic_ns()

        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunner
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID)
        # LLaMA models do not have a padding token, so we use the EOS token
        self.tokenizer.add_special_tokens(
            {"pad_token": self.tokenizer.eos_token}
        )
        # and then we add it from the left, to minimize impact on the output
        self.tokenizer.padding_side = "left"
        self.pad_id = self.tokenizer.pad_token_id
        self.end_id = self.tokenizer.eos_token_id

        runner_kwargs = dict(
            engine_dir=f"{LLAMA_ENGINE_DIR}",
            lora_dir=None,
            rank=tensorrt_llm.mpi_rank(),  # this will need to be adjusted to use multiple GPUs
        )

        self.llama_model = ModelRunner.from_dir(**runner_kwargs)

        self.init_duration_s = (time.monotonic_ns() - self.init_start) / 1e9
        print(
            f"{COLOR['HEADER']}🚀 Cold boot finished in {self.init_duration_s}s{COLOR['ENDC']}"
        )


    @modal.enter()
    def warmup(self):
        from whisper.normalizers import EnglishTextNormalizer
        from run import decode_dataset

        normalizer = EnglishTextNormalizer()
        results, _ = decode_dataset(  # warm up
            self.model,
            "hf-internal-testing/librispeech_asr_dummy",
            normalizer=normalizer,
            mel_filters_dir=self.assets_dir,
        )
        # To generate in chunks of at most 8:
        for i in range(0, len(results), 8):
            chunk = results[i:i+8]
            transcriptions = [" ".join(result[2]) for result in chunk]
            _ = self.generate.local(prompts=transcriptions)

        


    @modal.method()
    def generate(self, prompts: list[str], settings=None):
        """Generate responses to a batch of prompts, optionally with custom inference settings."""
        import time

        if settings is None or not settings:
            settings = dict(
                temperature=0.1,  # temperature 0 not allowed, so we set top_k to 1 to get the same effect
                top_k=1,
                stop_words_list=None,
                repetition_penalty=1.1,
            )

        settings[
            "max_new_tokens"
        ] = LLAMA_MAX_OUTPUT_LEN  # exceeding this will raise an error
        settings["end_id"] = self.end_id
        settings["pad_id"] = self.pad_id

        num_prompts = len(prompts)

        if num_prompts > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {num_prompts} exceeds maximum of {MAX_BATCH_SIZE}"
            )

        print(
            f"{COLOR['HEADER']}🚀 Generating completions for batch of size {num_prompts}...{COLOR['ENDC']}"
        )
        start = time.monotonic_ns()

        parsed_prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": """You are an AI assistant specializing in enhancing speech-to-text transcriptions. Given a transcription, your task is to:

- Repeat it as is, don't change the person being addressed
- Don't try to paraphrase or summarize, or complete the sentence
- Correct spelling and grammar errors
- Add appropriate punctuation
- Retain the original meaning and speaker's intent
- Preserve unique speech patterns or dialects
- Format the text for readability
- Fix mis-transcribed words or phrases

Do not add new information or significantly alter the content. Aim for a polished, accurate representation of the original speech. If certain words or phrases are unclear, indicate this with [unclear] brackets.
Include ONLY with actual the enhanced transcription in your response - no explanations, comments. Don't preface it with "Here is the enhanced transcription:" or similar."""},
                    {"role": "user", "content": prompt}
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]

        print(
            f"{COLOR['HEADER']}Parsed prompts:{COLOR['ENDC']}",
            *parsed_prompts,
            sep="\n\t",
        )

        inputs_t = self.tokenizer(
            parsed_prompts, return_tensors="pt", padding=True, truncation=False
        )["input_ids"]

        print(
            f"{COLOR['HEADER']}Input tensors:{COLOR['ENDC']}", inputs_t[:, :8]
        )

        try:
            outputs_t = self.llama_model.generate(inputs_t, **settings)
        except Exception as e:
            print(
                f"{COLOR['HEADER']}{COLOR['RED']}Error generating completions:{COLOR['ENDC']}",
                e,
            )  # Maximum input length (273) exceeds the engine or specified limit (256)
            return prompts  # return the original prompts

        outputs_text = self.tokenizer.batch_decode(
            outputs_t[:, 0]
        )  # only one output per input, so we index with 0

        responses = [
            extract_assistant_response(output_text)
            for output_text in outputs_text
        ]
        duration_s = (time.monotonic_ns() - start) / 1e9

        num_tokens = sum(
            map(lambda r: len(self.tokenizer.encode(r)), responses)
        )

        for prompt, response in zip(prompts, responses):
            print(
                f"{COLOR['HEADER']}{COLOR['GREEN']}{prompt}",
                f"\n{COLOR['BLUE']}{response}",
                "\n\n",
                sep=COLOR["ENDC"],
            )
            time.sleep(0.01)  # to avoid log truncation

        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Generated {num_tokens} tokens from {LLAMA_MODEL_ID} in {duration_s:.1f} seconds,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second for batch of size {num_prompts} on {GPU_CONFIG}.{COLOR['ENDC']}"
        )

        return responses

    @modal.asgi_app()
    def web(self):
        import io
        import librosa
        from fastapi import FastAPI, UploadFile, File
        from fastapi.responses import PlainTextResponse

        webapp = FastAPI()

        from run import decode_wav_file

        @webapp.get("/", response_class=PlainTextResponse)
        @webapp.get("/health", response_class=PlainTextResponse)
        async def health():
            server_logger.info("health check")
            return "OK"

        @webapp.post("/")
        @webapp.post("/predict")
        async def predict(
            file: Annotated[UploadFile, File()],
        ):
            contents = file.file.read()
            audio_data, _ = librosa.load(io.BytesIO(contents), sr=None)
            results, _ = decode_wav_file(
                audio_data,
                self.model,
                mel_filters_dir=self.assets_dir,
            )
            transcription = " ".join(results[0][2])
            llmed = self.generate.local(prompts=[transcription])
            return {
                "raw": transcription,
                "llama": llmed[0],
            }

        return webapp


@app.function(gpu=GPU_CONFIG)
def fn():
    print("hi")

@app.local_entrypoint()
def main():
    fn.remote()
