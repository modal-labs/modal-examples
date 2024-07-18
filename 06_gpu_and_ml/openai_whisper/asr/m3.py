import modal

from typing import Annotated
from modal import Image

INFERENCE_PRECISION = "float16"
WEIGHT_ONLY_PRECISION = "int8"
MAX_BEAM_WIDTH = 4
MAX_BATCH_SIZE = 8
checkpoint_dir = f"whisper_large_v3_weights_{WEIGHT_ONLY_PRECISION}"
output_dir= f"whisper_large_v3_{WEIGHT_ONLY_PRECISION}"

N_GPUS = 1
GPU_CONFIG = modal.gpu.A100(count=N_GPUS)

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
                --output_dir {checkpoint_dir}"
        ], gpu=GPU_CONFIG,
    )
    .run_commands(
        [
            f"trtllm-build --checkpoint_dir {checkpoint_dir}/encoder \
                  --output_dir {output_dir}/encoder \
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
            f"trtllm-build  --checkpoint_dir {checkpoint_dir}/decoder \
                  --output_dir {output_dir}/decoder \
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
    .pip_install("pydantic==1.10.11")
    .pip_install("librosa")
)

app = modal.App("faster-v2", image=image)

@app.cls(keep_warm=1, allow_concurrent_inputs=1, concurrency_limit=1, gpu=GPU_CONFIG)
class Model:
    @modal.enter()
    def enter(self):
        self.assets_dir = "/assets"
        from run import WhisperTRTLLM, decode_dataset

        self.model = WhisperTRTLLM(f"/{output_dir}", assets_dir=self.assets_dir)

        from whisper.normalizers import EnglishTextNormalizer
        normalizer = EnglishTextNormalizer()
        _, _ = decode_dataset(
            self.model,
            "hf-internal-testing/librispeech_asr_dummy",
            normalizer=normalizer,
            mel_filters_dir=self.assets_dir,
        )

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
            return results

        return webapp

@app.function(gpu=GPU_CONFIG)
def fn():
    print("hi")
