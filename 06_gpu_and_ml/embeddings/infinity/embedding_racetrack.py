
# # Modal Cookbook: Recipe for Inference Throughput Maximization
# In certain applications, the bottom line comes to throughput: process a set of inputs as fast as possible.
# Let's explore how to maximize throughput by using Modal on an embedding example, and see just how fast
# we can encode 10,000 images from the [wildflow sweet-coral dataset](https://huggingface.co/datasets/wildflow/sweet-corals "huggingface/wildflow/sweet-coral")
# using the [Infinity inference engine](https://github.com/michaelfeil/infinity "github/michaelfeil/infinity").

# ## BLUF (bottom line up front)
# We have found that setting concurrency to 2 and then maximizing the batchsize will maximize GPU utilization.
# If you then enable more containers, throughput goes through the roof.

## Setup
# Import everything we need for the locally-run Python (everything in our local_entrypoint function at the bottom).
from time import perf_counter
from pathlib import Path
from more_itertools import chunked
import csv, os
import numpy as np

import modal
from modal.volume import FileEntry
from PIL.Image import Image

# We will be running this example several times, and logging speeds to a local file specified here.
# Set local_logfile to `None` to ignore this.
local_logfile = None # Path("~/results/rt6.txt").expanduser()

app_name = 'embedding-racetrack'

# ## Key Parameters
# Key factors impacting throughput include batchsize, the amount of concurrency we allow for our app.
# * `batch_size` is a parameter passed to the [Infinity inference engine](https://github.com/michaelfeil/infinity "github/michaelfeil/infinity"), and it means the usual thing for machine learning inference: a group of images are processed through the neural network together.
# * `max_concurrency` sets the [@modal.concurrent(max_inputs:int) ](https://modal.com/docs/guide/concurrent-inputs#input-concurrency "Modal: input concurrency") argument for the inference app. 
# This takes advantage of the asynchronous nature of the Infinity embedding inference app.
# * `gpu` is a string specifying the GPU to be used. 
# * `max_containers` caps the number of containers allowed to spin-up.
# * `max_ims` caps the number of images used in this example
batch_size: int = 512
max_concurrency: int = 2
gpu: str = "H200"
max_containers: int = 2
max_ims: int = 20000
imload_workers = os.cpu_count()  # Number of CPU thread workers used to load images when creating the batch

# This should point to a model on huggingface that is supported by Infinity.
# Note that your specifically chosen model might require specialized imports when
# designing the image. This [OpenAI model](https://huggingface.co/openai/clip-vit-base-patch16 "OpenAI ViT")
# takes about 4-10s to load into memory.
MODEL_NAME = "openai/clip-vit-base-patch16"

# ## Data setup
# We use a [Modal Volume](https://modal.com/docs/guide/volumes#volumes "Modal.Volume") 
# to store the images we want to encode.
vol_name = "sweet-coral-db-20k"
vol_mnt = Path("/data")
vol = modal.Volume.from_name(vol_name)
def find_images_to_encode(image_cap:int = 1) -> list[FileEntry]:
    """
    You can modify this function to find an return a list of your image paths.
    """
        
    im_path_list = list(filter(lambda x: x.path.endswith(".jpg"),
                         vol.listdir('/data', recursive=True)))
    print(f"Found {len(im_path_list)} JPEGs, ", end='')

    # Optional: cutoff number of images for testing (set to -1 to encode all)
    if image_cap > 0:
        im_path_list = im_path_list[:min(len(im_path_list), image_cap)]
    return im_path_list

# ## Define the image
# Setup the image
simple_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(["infinity_emb[all]==0.0.76",   # for Infinity inference lib
                  "sentencepiece",               # for this particular chosen model
                  "more-itertools",              # for elegant list batching
                  "torchvision"])                # for fast image loading
    .env({"INFINITY_MODEL_ID": MODEL_NAME, "HF_HOME": "/data"})      # for Infinity inference lib and model caching
)

# Initialize the app
app = modal.App(app_name, image=simple_image, volumes={vol_mnt: vol})

# Imports inside the container
with simple_image.imports():
    from infinity_emb import AsyncEmbeddingEngine, EngineArgs
    from infinity_emb.primitives import Dtype, InferenceEngine
    from concurrent.futures import ThreadPoolExecutor
    from torchvision.io import read_image
    from torchvision.transforms.functional import to_pil_image

# ## Inference app
# Here we define an app.cls that wraps Infinity's AsyncEmbeddingEngine.
@app.cls(gpu=gpu, image=simple_image, volumes={vol_mnt: vol}, 
         timeout=24*60*60,
        #  min_containers=max_containers, 
         max_containers=max_containers)
@modal.concurrent(max_inputs=max_concurrency)
class InfinityModel:
    def make_batch(self, images: list[FileEntry])->list[Image]:
        # Convert to a list of paths
        images = images if isinstance(images, list) else [images]
        images = [getattr(im, "path", im) for im in images]

        def readim(impath):
            """ Read with torch, convert back to PIL for Infinity """
            return to_pil_image(read_image(str(vol_mnt / impath)))
        
        with ThreadPoolExecutor(max_workers=imload_workers) as executor:
            images = list(executor.map(readim, images))
    
        return images
    
    @modal.method()
    async def embed(self, images: list[str])->tuple[float, float]:
        # (0) Instantiate an Infinity Inference Engine
        st = perf_counter()
        self.model = AsyncEmbeddingEngine.from_args(EngineArgs(
                model_name_or_path=MODEL_NAME,
                batch_size=batch_size,
                model_warmup=False,
                engine=InferenceEngine.torch,
                dtype=Dtype.float16,
                device='cuda'
                ))
        await self.model.astart()
        model_elapsed = perf_counter() - st

        # (1) Load batch of image data
        st = perf_counter()
        images = self.make_batch(images)
        batch_elapsed = perf_counter() - st

        # (2) Encode the batch
        st = perf_counter()
        embedding, _ = await self.model.image_embed(images=images)
        embed_elapsed = perf_counter() - st

        # (3) Housekeeping
        print(f"Time to create model: {model_elapsed:.2f}s")
        print(f"Time to load batch: {batch_elapsed:.2f}s")
        print(f"Time to embed batch: {embed_elapsed:.2f}s")

        # (4) You may wish to return the embeddings themselves here
        return embed_elapsed, len(images)
    
    @modal.exit()
    async def exit(self) -> None:
        await self.model.astop()

# ## Local Entrypoint
# This code is run on your machine.
@app.local_entrypoint()
def backbone(expname:str=''):

    # (0) [optional] Check for local logging:
    do_logging = local_logfile is not None
    if do_logging:
        CSV_FILE = Path(local_logfile).with_suffix(".csv")
        local_logfile.parent.mkdir(parents=True, exist_ok=True)

    # (1) Init the model inference app
    start_time = perf_counter()
    embedder = InfinityModel()
    
    # (2) Catalog data: modify `find_images_to_encode` at the top of this file for your usecase.
    im_path_list = find_images_to_encode(max_ims)
    n_ims = len(im_path_list)
    print(f"using {n_ims}.")

    # (3) Embed batches via remote `map` call
    times, batchsizes = [], [] 
    for time, batchsize in embedder.embed.map(chunked(im_path_list, batch_size)):
        times.append(time)
        batchsizes.append(batchsize)
    
    # (4) Log
    if n_ims>0:
        total_duration = perf_counter() - start_time
        total_throughput = n_ims / total_duration
        embed_througputs = np.array([batchsize/time for batchsize,time in zip(batchsizes, times)])
        avg_throughput = embed_througputs.mean()
        std_throughput = embed_througputs.std()
        actual_concurrency = min(max_ims // batch_size, max_concurrency)

        log_msg = (
            f"simple_volume.py::{expname}::batch_size={batch_size}::n_ims={n_ims}::concurrency={max_concurrency}\n"
            f"\tTotal time:\t{total_duration/60:.2f} min\n"
            f"\tOverall throughput:\t{total_throughput:.2f} im/s\n"
            f"\tEmbedding-only throughput (avg):\t{avg_throughput:.2f} im/s\n"
            f"\tEmbedding-only throughput (std dev):\t{std_throughput:.2f} im/s\n"
        )

        print(log_msg)
        if do_logging:
            # append to the file (newline already in log_msg)
            with open(local_logfile, "a", encoding="utf-8") as outlog:
                outlog.write(log_msg)

            csv_exists = CSV_FILE.exists()
            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not csv_exists:
                    # write header
                    writer.writerow(["batch_size", "concurrency", "n_images", 'max_containers', 'gpu'
                                    'total_time', 'total_throughput', 
                                    'avg_instance_throughput', 'stddev_instance_throughput'])
                # write your row
                writer.writerow([batch_size, actual_concurrency, n_ims, max_containers, gpu,
                                total_duration, total_throughput, avg_throughput, std_throughput])


