import time
import logging
import modal

from modal import Image, asgi_app

from contextlib import asynccontextmanager

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


image = (
    Image.from_registry("nvidia/cuda:12.3.2-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "build-essential", 
        "libssl-dev", 
        "libffi-dev", 
        "libncurses5-dev", 
        "zlib1g-dev", 
        "libreadline-dev", 
        "libbz2-dev", 
        "libsqlite3-dev", 
        "wget", 
        "ffmpeg", 
        "git"
    )
    .pip_install(
        "accelerate==0.27.2",
        "torch==2.2.1",
        "fastapi==0.110.0",
        "pyannote-audio==3.1.1",
        "transformers==4.38.2",
        "numpy==1.26.4",
        "torchaudio==2.2.1",
        "uvicorn==0.27.1",
        "httpx==0.27.0",
        "python-multipart==0.0.9",
        "pydantic-settings==2.2.1",
        "pytest==8.1.1"
    )
    .run_commands(
        "rm /opt/nvidia/entrypoint.d/10-banner.sh",
        "rm /opt/nvidia/entrypoint.d/12-banner.sh",
        "rm /opt/nvidia/entrypoint.d/15-container-copyright.txt",
        "rm /opt/nvidia/entrypoint.d/30-container-license.txt",
    )
)

app = modal.App("diarization", image=image)

with image.imports():
    import torch

    from typing import Annotated
    from pydantic import Json
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.responses import PlainTextResponse
    from pyannote.audio import Pipeline
    from transformers import pipeline, AutoModelForCausalLM
    from huggingface_hub import HfApi

    from .utils.config import model_settings
    from .utils.validation_utils import validate_file, process_params
    from .utils.diarization_utils import diarize


@app.cls(keep_warm=1, concurrency_limit=1, allow_concurrent_inputs=1, gpu="a10g")
class Model:
    def __init__(self):
        self.models = {}
        pass

    @modal.enter()
    def enter(self):
        t0 = time.time()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"Using device: {device.type}")

        torch_dtype = torch.float32 if device.type == "cpu" else torch.float16

        # from pytorch 2.2 sdpa implements flash attention 2
        self.models["asr_pipeline"] = pipeline(
            "automatic-speech-recognition",
            model=model_settings.asr_model,
            torch_dtype=torch_dtype,
            device=device
        )
        logger.info(f"ASR model loaded: {model_settings.asr_model}")

        self.models["assistant_model"] = AutoModelForCausalLM.from_pretrained(
            model_settings.assistant_model,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ) if model_settings.assistant_model else None

        if self.models["assistant_model"]:
            self.models["assistant_model"].to(device)

        if model_settings.diarization_model:
            logger.info(f"Loading diarization model {model_settings.diarization_model}")
            # diarization pipeline doesn't raise if there is no token
            HfApi().whoami(model_settings.hf_token)
            self.models["diarization_pipeline"] = Pipeline.from_pretrained(
                checkpoint_path=model_settings.diarization_model,
                use_auth_token=model_settings.hf_token,
            )
            if self.models["diarization_pipeline"]:
                self.models["diarization_pipeline"].to(device)
        else:
            self.models["diarization_pipeline"] = None

        logger.info(f"Models loaded in {time.time() - t0:.2f} seconds")

    @asgi_app()
    def web(self):
        webapp = FastAPI()

        @webapp.get("/", response_class=PlainTextResponse)
        @webapp.get("/health", response_class=PlainTextResponse)
        async def health():
            logger.info("health check")
            return "OK"


        @webapp.post("/")
        @webapp.post("/predict")
        async def predict(
            file: Annotated[UploadFile, File()],
            parameters: Annotated[Json , Form()] = {}
        ):
            parameters = process_params(parameters)
            file = await validate_file(file)

            logger.info(f"inference parameters: {parameters}")

            generate_kwargs = {
                "task": parameters.task,
                "language": parameters.language,
                "assistant_model": self.models["assistant_model"] if parameters.assisted else None
            }

            try:
                logger.info("starting ASR pipeline")
                asr_outputs = self.models["asr_pipeline"](
                    file,
                    chunk_length_s=parameters.chunk_length_s,
                    batch_size=parameters.batch_size,
                    generate_kwargs=generate_kwargs,
                    return_timestamps=True,
                )
            except RuntimeError as e:
                logger.error(f"ASR inference error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"ASR inference error: {str(e)}")
            except Exception as e:
                logger.error(f"Unknown error during ASR inference: {e}")
                raise HTTPException(status_code=500, detail=f"Unknown error during ASR inference: {str(e)}")

            if self.models["diarization_pipeline"]:
                try:
                    transcript = diarize(self.models["diarization_pipeline"], file, parameters, asr_outputs)
                except RuntimeError as e:
                    logger.error(f"Diarization inference error: {str(e)}")
                    raise HTTPException(status_code=400, detail=f"Diarization inference error: {str(e)}")
                except Exception as e:
                    logger.error(f"Unknown error during diarization: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Unknown error during diarization: {str(e)}")
            else:
                transcript = []

            return {
                "transcript": transcript,
                "chunks": asr_outputs["chunks"],
                "text": asr_outputs["text"],
            }

        return webapp

