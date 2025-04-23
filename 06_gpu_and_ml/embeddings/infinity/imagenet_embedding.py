
# # Infinity-class COPIED FROM SUPERLINKED EXAMPLE - DO NOT RELEASE

# # COPIED FROM SUPERLINKED EXAMPLE - DO NOT RELEASE

# Imports
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os

import modal


###################
# Modify if needed
CPU = 1
MAX_CONCURRENT_REQUESTS = 1
GPU = "H200"
MAX_BATCH_SIZE = 2000  # 5000 would be too high for T4

###################
# Only modify if you are sure
SCALEDOWN_WINDOW = 600  # seconds
TIMEOUT = 600  # seconds
LOG_LEVEL = 10
APP_NAME = "embedding-overdrive"

#############
MODEL_NAMES = ["Alibaba-NLP/gte-large-en-v1.5", 
               "pySilver/marqo-fashionSigLIP-ST"]
ALLOCATED_MEMORY = CPU * MAX_BATCH_SIZE + 6000  # To handle images

# # Container Setup
# To compute embeddings, the inference app we are going to create
# needs access to the data and necessary python packages.
inf_image = (
    modal.Image.debian_slim(python_version="3.10")
    # For downloading data
    .pip_install(["roboflow~=1.1.37", "opencv-python~=4.10.0"])
    # For infinity encoder inference
    .pip_install(["infinity_emb[all]==0.0.76", "hf_xet==1.0.3",
                  "ftfy==6.3.1", "open_clip_torch==2.32.0"])
    .env({"INFINITY_PORT": "7997", "INFINITY_MODEL_ID": ";".join(MODEL_NAMES)})
)

# We also create a persistent [Volume](https://modal.com/docs/guide/volumes) for storing datasets, trained weights, and inference outputs.

data_volume = modal.Volume.from_name(APP_NAME, create_if_missing=True)
volume_path = Path("/root") / "data" #  the path to the volume from within the container

# We attach both of these to a Modal [App](https://modal.com/docs/guide/apps).
app = modal.App(APP_NAME, 
                image=inf_image, 
                volumes={volume_path: data_volume})

# # Dataset Setup
# COPIED FROM CHARLES' ROBOFLOW DEMO: 
# TODO: how fan out the dataloader and stream to batched app

@dataclass
class DatasetConfig:
    """Information required to download a dataset from Roboflow."""

    workspace_id: str
    project_id: str
    version: int
    format: str
    target_class: str

    @property
    def id(self) -> str:
        return f"{self.workspace_id}/{self.project_id}/{self.version}"


@app.function(
    secrets=[
        modal.Secret.from_name("roboflow-api-key", required_keys=["ROBOFLOW_API_KEY"])
    ]
)
def download_dataset(config: DatasetConfig):
    import os

    from roboflow import Roboflow

    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = (
        rf.workspace(config.workspace_id)
        .project(config.project_id)
        .version(config.version)
    )
    dataset_dir = volume_path / "dataset" / config.id
    project.download(config.format, location=str(dataset_dir))


@app.function()
def read_image(image_path: str):
    import cv2             #TODO: better to import in context?

    source = cv2.imread(image_path)
    return source
########################################################################
########################################################################
########################################################################


@app.cls(
    gpu=GPU,
    cpu=CPU,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT,
    memory=ALLOCATED_MEMORY,
    max_containers=1,
    min_containers=1,
)
@modal.concurrent(max_inputs=1)  # we do not allow concurrency for 1 container
class Infinity:

    @modal.enter()
    async def init_model_server(self) -> None:

        import typing
        import subprocess

        from infinity_emb import AsyncEngineArray, EngineArgs
        from infinity_emb.primitives import Dtype, InferenceEngine
        from infinity_emb.transformer.abstract import BaseEmbedder

        # Start server
        exit_code = subprocess.Popen("infinity_emb v2 --preload-only", shell=True).wait()
        assert exit_code == 0, f"Failed to download models. Exit code: {exit_code}"

        # Initialize models
        args = [
            EngineArgs(
                model_name_or_path=model_name,
                batch_size=self.get_max_batch_size(),
                model_warmup=False,
                engine=InferenceEngine.torch,
                dtype=Dtype.float16,
            )
            for model_name in MODEL_NAMES
        ]
        self._engine_array = AsyncEngineArray.from_args(args)
        await self._engine_array.astart()
        
        self.embedders = {
            model_name: typing.cast(BaseEmbedder, 
                                    self._engine_array[model_name]._model_replicas[0])
            for model_name in MODEL_NAMES
        }

    def embed(self, model_name, images: list[bytes]) -> list[list[float]]:
        from PIL import Image as PilImage
        import io
        pre_encoded = self.embedders[model_name].encode_pre(processed_inputs)
        core_encoded = self.embedders[model_name].encode_core(pre_encoded)
        return self.embedders[model_name].encode_post(core_encoded)

    @modal.method()
        def singleim(self, batch_dir: os.Pathlike):
        batchdir = Path(batch_dir)
        for image in read_image.map(batchdir.rglob("*.png")):

    @modal.method()
    async def batch_embed(self, 
                          batch: list[str | bytes], 
                          model: str) -> list[list[float]]:
        from concurrent.futures import ThreadPoolExecutor
        from math import ceil

        batch_size = ceil(len(sentences) / CPU)
        batches = [sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)]
        worker_count = len(batches)

        if worker_count == 1:
            return do_embedding(sentences)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            batched_embeddings = executor.map(do_embedding, batches)

        return [embedding for batch_results in batched_embeddings for embedding in batch_results]

    @modal.method()
    async def get_max_batch_size(self) -> int:
        return MAX_BATCH_SIZE

    @modal.exit()
    async def exit(self) -> None:
        await self._engine_array.astop()


@app.local_entrypoint()
def main():
    deployment = Infinity()

    modelname = "Alibaba-NLP/gte-large-en-v1.5" #, "pySilver/marqo-fashionSigLIP-ST"]

    outputs=deployment.embed.remote(modelname, sentences=["hello world"])
    
    print(outputs)
    
    # embeddings_2 = deployment.image_embed.remote(
    #     urls=["http://images.cocodataset.org/val2017/000000039769.jpg"],
    #     model=model_id[0],
    # )

    # rerankings_1 = deployment.rerank.remote(
    #     query="Where is Paris?",
    #     docs=["Paris is the capital of France.", "Berlin is a city in Europe."],
    #     model=model_id[2],
    # )

    # classifications_1 = deployment.classify.remote(
    #     sentences=["I feel great today!"], model=model_id[3]
    # )

    # print(
    #     "Success, all tasks submitted! Embeddings:",
    #     embeddings_1[0].shape,
    #     embeddings_2[0].shape,
    #     "Rerankings:",
    #     rerankings_1,
    #     "Classifications:",
    #     classifications_1,
    # )