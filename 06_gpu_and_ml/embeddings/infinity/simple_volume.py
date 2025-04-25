
# # Modal Cookbook: Recipe for Inference Throughput Maximization
# In certain applications, the bottom line comes to throughput: process a set of inputs as fast as possible.
# Let's explore how to maximize throughput by using Modal on an embedding example, and see just how fast
# we can encode the [wildflow sweet-coral dataset](https://huggingface.co/datasets/wildflow/sweet-corals "huggingface/wildflow/sweet-coral")
# using the [Infinity inference engine](https://github.com/michaelfeil/infinity "github/michaelfeil/infinity").

# Import everything we need for the locally-run Python (everything in our local_entrypoint function at the bottom).
from time import time as clock
from pathlib import Path
from more_itertools import chunked
import csv

import modal

# We will be running this example several times, and logging speeds to a local file specified here.
local_logfile = Path("~/results/rt5.txt").expanduser()
CSV_FILE = Path(local_logfile).with_suffix(".csv")
local_logfile.parent.mkdir(parents=True, exist_ok=True)


# ## Key Parameters
# Key factors impacting throughput include batchsize, the amount of concurrency we allow for our app.
# * `batch_size` is a parameter passed to the [Infinity inference engine](https://github.com/michaelfeil/infinity "github/michaelfeil/infinity"), and it means the usual thing for machine learning inference: a group of images are processed through the neural network together.
# * `max_concurrency` sets the [@modal.concurrent(max_inputs:int) ](https://modal.com/docs/guide/concurrent-inputs#input-concurrency "Modal: input concurrency") argument for the inference app. 
# This takes advantage of the asynchronous nature of the Infinity embedding inference app.
# * `gpu` is a string specifying the GPU to be used. 
batch_size: int = 10
max_concurrency: int = 1000
gpu: str = "H200"
max_containers: int = 1
# This `max_ims` parameter simply caps the total number of images that are parsed (for testing/debugging).
# Set to -1 to parse all.
max_ims: int = 1000

# This should point to a model on huggingface that is supported by Infinity.
# Note that your specifically chosen model might require specialized imports when
# designing the image!
MODEL_NAME = "openai/clip-vit-base-patch16"

# ## Define the image and data volume
# Setup the image
simple_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(["infinity_emb[all]==0.0.76",   # for Infinity inference lib
                  "sentencepiece",               # for this particular chosen model
                  "more-itertools"])             # for elegant list batching
    .env({"INFINITY_MODEL_ID": MODEL_NAME})      # for Infinity inference lib
)

# Setup a [Modal Volume](https://modal.com/docs/guide/volumes#volumes "Modal.Volume") containing all of the images we want to encode.
vol_name = "sweet-coral-db-20k"
vol_mnt = Path("/data")
vol = modal.Volume.from_name(vol_name)

# Initialize the app
app = modal.App('vol-simple-infinity', image=simple_image, volumes={vol_mnt: vol})

# Imports inside the container
with simple_image.imports():
    from infinity_emb import AsyncEmbeddingEngine, EngineArgs
    from infinity_emb.primitives import Dtype, InferenceEngine
    from PIL import Image

# Here we define an app.cls that wraps Infinity's AsyncEmbeddingEngine.
@app.cls(gpu=gpu, image=simple_image, volumes={vol_mnt: vol}, 
         timeout=24*60*60,
         min_containers=max_containers, max_containers=max_containers)
@modal.concurrent(max_inputs=max_concurrency)
class InfinityModel:

    # The enter* decorator
    @modal.enter()
    async def enter(self):
        self.model = AsyncEmbeddingEngine.from_args(EngineArgs(
                model_name_or_path=MODEL_NAME,
                batch_size=batch_size,
                model_warmup=False,
                engine=InferenceEngine.torch,
                dtype=Dtype.float16,
                ))
        await self.model.astart()
    # TODO: get the ecit funcvtion

    @modal.method()
    async def embed(self, images: list[str])->list[int]:
        # Convert to a list of volume paths
        start = clock()

        images = images if isinstance(images, list) else [images]
        images = [getattr(im, "path", im) for im in images]
        # File paths to images
        images = [Image.open(vol_mnt / impath) for impath in images]

        embedding, _ = await self.model.image_embed(images=images)
        return clock()-start

@app.local_entrypoint()
def backbone(expname:str=''):
    st=clock()

    # Init the model inference app
    embedder = InfinityModel()
    startup_dur = clock() - st
    print(f"Took {startup_dur}s to start Infinity Engine.")
    
    # Catalog data
    im_path_list = list(filter(lambda x: x.path.endswith(".jpg"),
                         vol.listdir('/data', recursive=True)))
    print(f"Found {len(im_path_list)} JPEGs, ", end='')

    # Optional: cutoff number of images for testing (set to -1 to encode all)
    if max_ims > 0:
        im_path_list = im_path_list[:min(len(im_path_list), max_ims)]
    n_ims = len(im_path_list)
    print(f"using {n_ims}.")

    # Embed batches via remote `map` call
    throughputs=[]
    for thru_put in embedder.embed.map(chunked(im_path_list, batch_size)):
        throughputs.append(thru_put)
    print(throughputs[0])
    # Time it!
    total_duration = clock() - st

    total_embed_time = sum(throughputs)
    total_throughput = n_ims / total_embed_time

    # Log
    if n_ims>0:
        log_msg = (
            f"simple_volume.py::{expname}::batch_size={batch_size}::n_ims={n_ims}::concurrency={max_concurrency}\n"
            f"\tTotal time:\t{total_duration/60:.2f} min\n"
            f"\tOverall throughput:\t{n_ims/total_duration:.2f} im/s\n"
            f"\tEmbedding-only throughput (avg):\t{total_embed_time:.2f} im/s\n"
        )

        # append to the file (newline already in log_msg)
        with open(local_logfile, "a", encoding="utf-8") as outlog:
            outlog.write(log_msg)

        csv_exists = CSV_FILE.exists()
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not csv_exists:
                # write header
                writer.writerow(["batch_size", "concurrency", "n_images", 'total_time', 'total_embed_time', 'max_containers', 'total_thru_put'])
            # write your row
            writer.writerow([batch_size, max_concurrency, n_ims, total_duration, total_embed_time, max_containers, n_ims/total_duration])
    # Store the codes (?)

