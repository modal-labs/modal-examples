
# # Infinity-class COPIED FROM SUPERLINKED EXAMPLE - DO NOT RELEASE

# # COPIED FROM SUPERLINKED EXAMPLE - DO NOT RELEASE

# Imports
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os, time
from abc import abstractmethod
import modal
from itertools import islice


###################
# Modify if needed
CPU = 1
MAX_CONCURRENT_REQUESTS = 1 # 500 on Infinity, 1 on superlink
GPU = "H200"
MAX_BATCH_SIZE = 2000  # 5000 would be too high for T4
BATCHED_WAIT = 1000 # ms
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
    # TODO: better to import in image context?
    import cv2             

    source = cv2.imread(image_path)
    return source
########################################################################
########################################################################
########################################################################

# # FROM SUPERLINK
# @app.cls(
#     gpu=GPU,
#     cpu=CPU,
#     scaledown_window=SCALEDOWN_WINDOW,
#     timeout=TIMEOUT,
#     memory=ALLOCATED_MEMORY,
#     max_containers=1,
#     min_containers=1,
# )

class InfinityBaseClass:

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
    
    @modal.exit()
    async def exit(self) -> None:
        await self._engine_array.astop()

    def embed(self, model_name, images: list[bytes]) -> list[list[float]]:
        pre_encoded = self.embedders[model_name].encode_pre(images)
        core_encoded = self.embedders[model_name].encode_core(pre_encoded)
        return self.embedders[model_name].encode_post(core_encoded)

@app.cls(gpu=GPU)
class InfinityStream(InfinityBaseClass):
    @modal.method()
    def stream_singleims(self, data_dir: os.PathLike):
        import time
        start = time.monotonic_ns()
        count=0
        
        batchdir = Path(data_dir)
        for image in read_image.map(batchdir.rglob("*.png")):
            self.embed([image])
            count+=1

        sec = (time.monotonic_ns() - start) / 1e9
        print(f'InfinityStream:\n'
              f'\t{count}ims@{len(count)/sec:.1}im/s ({sec/60:.1f}min)')


@app.cls(gpu=GPU, allow_concurrent_inputs=MAX_CONCURRENT_REQUESTS)
class InfinityConcurrent(InfinityBaseClass):
    """
    A la SuperLink's solution
    """

    @modal.method()
    async def batch_embed(self, data_dir: os.PathLike, batch_size=1):
        from concurrent.futures import ThreadPoolExecutor
        from math import ceil
        import time
        start = time.monotonic_ns()
        
        ims = list(Path(data_dir).rglob('.png'))
        batches = [ims[i : i + batch_size] 
                   for i in range(len(ims), batch_size)]
        
        # ???
        worker_count = len(batches)
        if worker_count == 1:
            return self.embed(ims)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            batched_embeddings = executor.map(self.embed, batches)

        sec = (time.monotonic_ns() - start) / 1e9
        print(f'InfinityConcurrent (batch_size={batch_size}):\n'
              f'\t{len(ims)}ims@{len(ims)/sec:.1}im/s ({sec/60:.1f}min)\n'
              f'\t{len(batches)}batches@{len(batches)/sec:.4}s each')

        # return [embedding for batch_results in batched_embeddings for embedding in batch_results]

@app.cls(gpu=GPU, allow_concurrent_inputs=MAX_CONCURRENT_REQUESTS)
class InfinityBatched(InfinityBaseClass):
    """
    Best of both worlds?...
    """
    def __init__(self, batch_size=1):
        # Grab the *unbound* function, decorate it, then bind it back:
        self.bsz = batch_size
        self.total_embed_time = 0
        self.n_batches = 0
        self.n_ims = 0
        raw_fn = self.batch_embed.__func__  
        decorated = modal.batched(
            max_batch_size=self.bsz,
            wait_ms=BATCHED_WAIT
        )(raw_fn)
        # bind `decorated(self, …)` as an *instance* method
        self.batch_embed = decorated.__get__(self, type(self))

    async def batch_embed(self, im_list: list):
        from concurrent.futures import ThreadPoolExecutor
        from math import ceil
        import time
        start = time.monotonic_ns()
        
        self.embed(im_list)

        self.total_embed_time += (time.monotonic_ns() - start) / 1e9
        self.n_batches+=1
        self.n_ims += len(im_list)
        


@app.local_entrypoint()
def main():
    quick_test: bool = True
    test_images = volume.listdir(
        str(Path("dataset") / dataset.id / "test" / "images")
    )
    if quick_test:
        test_images=test_images[:100]

    #######################################################
    print('---------------------')
    st = time.monotonic_ns()
    embedder = InfinityStream()
    embedder.stream_singleims.remote(data_path)
    print(f"Total stream method time: {(time.time()-st)/min:.2}min")
    del embedder

    # A la SUPERLINK example
    for batch_size in [2000, 5000, 10000, 30000, len(test_images)]:
        
        try:
            if batch_size < len(test_images):
                continue
            #######################################################
            print('---------------------')
            st = time.monotonic_ns()
            embedder = InfinityConcurrent()
            embedder.batch_embed.remote(data_path, batch_size=batch_size)
            print(f"Total concurrent method time (bsz={batch_size}): {(time.time()-st)/min:.2}min")
            del embedder

            #######################################################
            # multi node approach            
            print('---------------------')
            st = time.monotonic_ns()
            embedder = InfinityBatched(batch_size)

            for image in read_image.map(data_path.rglob("*.png")):
                # forward to the queue?
                embedder.batch_embed.remote(image)

            print(f'InfinityBatched (batch_size={batch_size}):\n'
                    f'\t{embedder.n_ims}ims@{embedder.n_ims/embedder.total_embed_time:.1}im/s ({embedder.total_embed_time/60:.1f}min)\n'
                    f'\t{embedder.n_batches}batches@{embedder.n_batches/sec:.4}s each')
            print(f"Total Batched method time (bsz={batch_size}): {(time.time()-st)/min:.2}min")
            del embedder
        except Exception as err:
            print(f"Got err when trying batch approach with "
                  f"batchsize:{batch_size}\n\t{err}")



    modelname = "Alibaba-NLP/gte-large-en-v1.5" #, "pySilver/marqo-fashionSigLIP-ST"]

    outputs=deployment.embed.remote(modelname, sentences=["hello world"])
    
    print(outputs)