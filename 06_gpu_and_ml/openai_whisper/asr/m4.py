import modal

from typing import Annotated
from modal import Image
import logging

INFERENCE_PRECISION = "float16"
WEIGHT_ONLY_PRECISION = "int8"
MAX_BEAM_WIDTH = 4
MAX_BATCH_SIZE = 8
WHISPER_OUTPUT_DIR = f"whisper_large_v3_weights_{WEIGHT_ONLY_PRECISION}"
WHISPER_CHECKPOINT_DIR= f"whisper_large_v3_{WEIGHT_ONLY_PRECISION}"


LLAMA_MODEL_DIR = "/root/model/model_input"
LLAMA_MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
LLAMA_MODEL_REVISION = "b1532e4dee724d9ba63fe17496f298254d87ca64"  # pin model revisions to prevent unexpected changes!
LLAMA_CHECKPOINT_SCRIPT_URL = f"https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/examples/llama/convert_checkpoint.py"
LLAMA_CKPT_DIR = "llama_3_8b_weights"
LLAMA_OUTPUT_DIR = "llama_3_8b"
MAX_INPUT_LEN = 256
SIZE_ARGS = f"--max_batch_size={MAX_BATCH_SIZE} --max_input_len={MAX_INPUT_LEN}"
DTYPE = "float16"
PLUGIN_ARGS = f"--gemm_plugin={DTYPE} --gpt_attention_plugin={DTYPE}"



N_GPUS = 1
GPU_CONFIG = modal.gpu.A100(count=N_GPUS)
DTYPE = "float16"

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger

logger = setup_logger()

def download_llama_model():
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
    .run_commands([  # get rid of CUDA banner
        "rm /opt/nvidia/entrypoint.d/10-banner.sh",
        "rm /opt/nvidia/entrypoint.d/12-banner.sh",
        "rm /opt/nvidia/entrypoint.d/15-container-copyright.txt",
        "rm /opt/nvidia/entrypoint.d/30-container-license.txt",
    ])
    .run_commands([
        "wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken",
        "wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz",
        "wget --directory-prefix=assets https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav",
        "wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
        
        "wget --directory-prefix=whisper_scripts https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/examples/whisper/convert_checkpoint.py",
        "wget --directory-prefix=whisper_scripts https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/examples/whisper/run.py",
        "wget --directory-prefix=whisper_scripts https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/examples/whisper/requirements.txt",
        "wget --directory-prefix=whisper_scripts https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/examples/whisper/tokenizer.py",
        
        f"wget --directory-prefix=llama_scripts {LLAMA_CHECKPOINT_SCRIPT_URL}",
        "pip install -r whisper_scripts/requirements.txt"
    ])
    .pip_install(  # add utilities for downloading the model
        "hf-transfer==0.1.6",
        "requests~=2.31.0",
    )
    .run_function(  # download the model
        download_llama_model,
        timeout=20 * 60,
    )
    # .copy_local_file("convert_checkpoint.py")
    .run_commands(
        [
            f"python whisper_scripts/convert_checkpoint.py \
                --use_weight_only \
                --weight_only_precision {WEIGHT_ONLY_PRECISION} \
                --output_dir {WHISPER_CHECKPOINT_DIR}"
        ], gpu=GPU_CONFIG,
    )
    .run_commands(
        [
            f"python llama_scripts/convert_checkpoint.py --model_dir={LLAMA_MODEL_DIR} --output_dir={LLAMA_CKPT_DIR}"
            + f" --tp_size={N_GPUS} --dtype={DTYPE}"
        ], gpu=GPU_CONFIG
    )
    .run_commands(
        [
            f"trtllm-build --checkpoint_dir {WHISPER_CHECKPOINT_DIR}/encoder \
                  --output_dir {WHISPER_OUTPUT_DIR}/encoder \
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
            f"trtllm-build  --checkpoint_dir {WHISPER_CHECKPOINT_DIR}/decoder \
                  --output_dir {WHISPER_OUTPUT_DIR}/decoder \
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
    .run_commands(  # takes ~5 minutes
        [
            f"trtllm-build --checkpoint_dir {LLAMA_CKPT_DIR} --output_dir {LLAMA_OUTPUT_DIR}"
            + f" --tp_size={N_GPUS} --workers={N_GPUS}"
            + f" {SIZE_ARGS}"
            + f" {PLUGIN_ARGS}"
        ],
        gpu=GPU_CONFIG,  # TRT-LLM compilation is GPU-specific, so make sure this matches production!
    ).env(  # show more log information from the inference engine
        {"TLLM_LOG_LEVEL": "INFO"}
    )
    # .copy_local_file("run.py", "/root/run.py")
    # .copy_local_file("tokenizer.py", "/root/tokenizer.py")
    # .copy_local_file("whisper_utils.py", "/root/whisper_utils.py")
    .pip_install("pydantic==1.10.11")
    .pip_install("librosa")
    .run_commands(["wget --directory-prefix=whisper_scripts https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/examples/whisper/whisper_utils.py"])
)

app = modal.App("faster-v2", image=image)

@app.cls(keep_warm=1, allow_concurrent_inputs=1, concurrency_limit=1, gpu=GPU_CONFIG)
class Model:
    @modal.enter()
    def enter(self):
        self.assets_dir = "/assets"
        import sys
        sys.path.append('/whisper_scripts')
        # import importlib.util

        from run import WhisperTRTLLM, decode_dataset
        self.whisper_model = WhisperTRTLLM(f"/{WHISPER_OUTPUT_DIR}", assets_dir=self.assets_dir)

        from whisper.normalizers import EnglishTextNormalizer
        normalizer = EnglishTextNormalizer()
        _, _ = decode_dataset(
            self.whisper_model,
            "hf-internal-testing/librispeech_asr_dummy",
            normalizer=normalizer,
            mel_filters_dir=self.assets_dir,
        )


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
            engine_dir=f"/{LLAMA_OUTPUT_DIR}",
            lora_dir=None,
            rank=tensorrt_llm.mpi_rank(),  # this will need to be adjusted to use multiple GPUs
        )

        self.llama_model = ModelRunner.from_dir(**runner_kwargs)

 

    @modal.asgi_app()
    def web(self):
        import io
        import librosa
        from fastapi import FastAPI, UploadFile, File
        from fastapi.responses import PlainTextResponse

        webapp = FastAPI()
        import sys
        sys.path.append('/whisper_scripts')
        from run import decode_wav_file

        @webapp.get("/", response_class=PlainTextResponse)
        @webapp.get("/health", response_class=PlainTextResponse)
        async def health():
            logger.info("health check")
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
                self.whisper_model,
                mel_filters_dir=self.assets_dir,
            )
            result_sentence = results[0][2] 

            settings = dict(
                temperature=0.1,  # temperature 0 not allowed, so we set top_k to 1 to get the same effect
                top_k=1,
                stop_words_list=None,
                repetition_penalty=1.1,
            )

            settings["end_id"] = self.end_id
            settings["pad_id"] = self.pad_id

            parsed_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"Please translate the following text into Spanish: {result_sentence}"}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            
            inputs_t = self.tokenizer(
                [parsed_prompt], return_tensors="pt", padding=True, truncation=False
            )["input_ids"]
            
            outputs_t = self.llama_model.generate(inputs_t, **settings)

            outputs_text = self.tokenizer.batch_decode(
                    outputs_t[:, 0]
                )  # only one output per input, so we index with 0

            responses = [
                extract_assistant_response(output_text)
                for output_text in outputs_text
            ]
            return responses[0]

        return webapp

@app.function(image = image, gpu=GPU_CONFIG)
def fn():
    print("hi")


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