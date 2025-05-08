import asyncio
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Iterator, List, Sequence, Tuple

import modal

vol_name = "example-embedding-data"
vol_mnt = Path("/data")
TH_CACHE_DIR = vol_mnt / "model-compile-cache"
# If this is set to something other than `None`, will cancel max_containers parameters below
buffer_containers = None  # 1

hf_secret = modal.Secret.from_name("huggingface-secret")
data_volume = modal.Volume.from_name(vol_name, create_if_missing=True)
th_compile_kwargs = {"mode": "reduce-overhead", "fullgraph": True, "dynamic": False}
app_cfg = {"buffer_containers": buffer_containers} if buffer_containers else {}

# ### Define the image
th_compile_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            # TODO: some of this required for the full embedding demo with data setup
            "datasets",  # for huggingface data download
            "hf_transfer",  # for fast huggingface data download
            "tqdm",  # progress bar for dataset download
            "torch",  # torch.compile
            "transformers",  # CLIPVisionModel etc.
            "torchvision",  # for fast image loading
        ]
    )
    .env(
        {
            # For fast HuggingFace model and data caching and download in our Volume
            "HF_HOME": vol_mnt.as_posix(),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Enables speedy caching across containers
            "TORCHINDUCTOR_CACHE_DIR": TH_CACHE_DIR.as_posix(),
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
        }
    )
)

# Initialize the app
app = modal.App(
    "example-multiprocessing-embedder",
    image=th_compile_image,
    volumes={vol_mnt: data_volume},
    secrets=[hf_secret],
)

# Imports inside the container
with th_compile_image.imports():
    import torch
    from torch import Tensor
    from torchvision.io import read_image
    from transformers import CLIPImageProcessorFast, CLIPVisionConfig, CLIPVisionModel


# ## Data
# @dataclass(frozen=True, slots=True)
class _WorkerCfg:
    """Serialisable package for workers"""

    def __init__(
        self,
        model_config,
        state_dict,
        preprocessor,
        compile_cache,
        device_id,
        input_shape,
    ):
        self.model_config = model_config
        self.state_dict = state_dict
        self.preprocessor = preprocessor
        self.compile_cache = compile_cache
        self.device_id = device_id
        self.input_shape = input_shape


def chunked(seq: list[os.PathLike], subseq_size: int) -> Iterator[list[os.PathLike]]:
    """
    Helper function that chunks a sequence into subsequences of length `subseq_size`.
    """
    for i in range(0, len(seq), subseq_size):
        yield seq[i : i + subseq_size]


# ## Worker (Process)


def _fmt_msg(rank, time_in_queue, cuda_time, inf_time, cudactxptr):
    msg = f"Process {rank} (PID={os.getpid()}, ctx=0c{cudactxptr}"
    msg += f"\n\ttime in queue: {time_in_queue:.2E}"
    msg += f"\n\tto-cuda time: {cuda_time:.2E}"
    msg += f"\n\tinference time: {inf_time:.2E}"
    return msg


def _worker_loop(
    rank: int,
    pinned_bufs: List[torch.Tensor],
    free_q: "torch.multiprocessing.SimpleQueue",
    ready_q: "torch.multiprocessing.SimpleQueue",
    out_q: "torch.multiprocessing.SimpleQueue",
    cfg: "_WorkerCfg",
    verbose: bool = False,
) -> None:
    """
    Single‑GPU worker executed in a subprocess

    Parameters
    ----------
    rank : int
        Worker index for logging only.
    pinned_bufs : List[Tensor]
        Shared pinned‑host ring buffer.
    free_q / ready_q / out_q
        IPC queues coordinating slots & outputs.
    cfg : _WorkerCfg
        Frozen dataclass with model weights & hyper‑parameters.
    """
    with torch.no_grad():
        #########
        ## Printing stuff──────────────────────────────────
        import ctypes
        from ctypes.util import find_library

        from torch.compiler import load_cache_artifacts
        from torch.compiler._cache import CacheInfo
        from torch.serialization import safe_globals

        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        if major > 8:
            torch.set_float32_matmul_precision("high")

        def cuda_context_ptr():
            CUcontext = ctypes.c_void_p
            ctx = CUcontext()
            libcuda = ctypes.CDLL(find_library("cuda"))
            libcuda.cuInit(0)
            libcuda.cuCtxGetCurrent(ctypes.byref(ctx))
            return ctx.value  # integer handle

        # ─── per‑process CUDA initialisation ──────────────────────────────────
        st = perf_counter()
        torch.cuda.set_device(cfg.device_id)
        stream = torch.cuda.Stream()

        # Make sure cache is loaded
        with safe_globals([CacheInfo]):
            load_cache_artifacts(cfg.compile_cache.read_bytes())

        # Instantiate model (1-2s)
        model = CLIPVisionModel(cfg.model_config).eval().cuda()
        model.load_state_dict(cfg.state_dict)
        # Compile (2-4s)
        model = torch.compile(model, **th_compile_kwargs)
        model(
            **cfg.preprocessor(
                images=torch.randn(*cfg.input_shape),
                device=model.device,
                return_tensors="pt",
            )
        )

        # TODO: unsure how necessary this is but seemed to reduce initial post time
        with torch.cuda.stream(stream):
            for buf in pinned_bufs:
                _ = pinned_bufs[0].cuda(non_blocking=True)  # launch async H2D
        torch.cuda.current_stream().wait_stream(stream)  # ensure it finishes
        torch.cuda.synchronize()  # belt-and-suspenders

        if verbose:
            print(
                f"Worker{rank}: loaded and compiled model in {perf_counter() - st:2E}"
            )

        # ───── loop ──────────────────────────────────────────────────────────
        while True:
            # Look out for a batch in the ready_q
            item = ready_q.get()
            if item is None:
                break
            # `slot` is just the index of the pinned buffer
            slot, t_post = item
            time_in_queue = perf_counter() - t_post

            # ─── H2D async copy ────────────────────────────────────────────
            t_cpu2gpu = perf_counter()
            with torch.cuda.stream(stream):
                batch_gpu = pinned_bufs[slot].cuda(non_blocking=True)
            torch.cuda.current_stream().wait_stream(stream)
            cuda_time = perf_counter() - t_cpu2gpu

            # Pinned buffer is free now
            free_q.put(slot)

            # ─── forward pass ─────────────────────────────────────────────
            t_inf_start = perf_counter()
            embed = model(
                **cfg.preprocessor(images=batch_gpu, return_tensors="pt")
            ).pooler_output
            inf_time = perf_counter() - t_inf_start
            # Output the CPU embedding ptr, a message, some times
            out_q.put(
                (
                    embed.cpu(),
                    _fmt_msg(
                        rank, time_in_queue, cuda_time, inf_time, cuda_context_ptr()
                    ),
                    inf_time,
                )
            )


# ## Inference App
@app.cls(
    image=th_compile_image,
    volumes={vol_mnt: data_volume},
    timeout=5 * 60,  # 5min timeout for large models + batches
    cpu=8,
    memory=20 * 1024,  # MB -> GB
    include_source=True,
    **app_cfg,
)
class MPEngine:
    model_name: str = modal.parameter()
    batch_size: int = modal.parameter(default=100)
    n_engines: int = modal.parameter(default=1)
    concurrency: int = modal.parameter(default=1)
    im_c: int = modal.parameter(default=3)
    im_h: int = modal.parameter(default=224)
    im_w: int = modal.parameter(default=224)
    threads_per_core: int = modal.parameter(default=4)
    verbose_inference: bool = modal.parameter(default=False)
    # Cannot currently gracefully set ENV vars from local_entrypoint
    cache_dir: Path = TH_CACHE_DIR
    # For logging
    name: str = "MPEngine"

    def get_compile_bytes(self, model, input_shape, preprocessor, compile_cache):
        """
        Check for compiled model cache; if nothing found, trigger re-trace
        in principal should only happen once per container but it's hard to
        "convince" torchInductor no further compilation is needed...
        """
        if compile_cache.exists():
            artifact_bytes = compile_cache.read_bytes()
            opt_msg = ""
        else:
            print("Parent compilation...", end="")
            # Compile once to save the bytes
            model = torch.compile(model, **th_compile_kwargs)
            # Force trace
            model(**preprocessor(images=torch.randn(*input_shape), return_tensors="pt"))
            # Write the bytes
            artifact_bytes, _ = torch.compiler.save_cache_artifacts()
            compile_cache.parent.mkdir(parents=True, exist_ok=True)
            compile_cache.write_bytes(artifact_bytes)
            print("done.")
            opt_msg = " and re-compile the"
        out_msg = f"\n\ttime to load{opt_msg} model in Parent: "
        return artifact_bytes, out_msg

    @modal.enter()
    async def init_engines(self):
        import torch.multiprocessing as mp

        torch.set_grad_enabled(False)

        msg = "New container!"
        # (0) Setup

        mp.set_start_method("spawn", force=True)
        ctx = mp.get_context("spawn")

        # This makes sure n-th container finds the torch.compile cache created by the first one
        data_volume.reload()

        # This is where we will cache torch.compile artifacts
        compile_cache: Path = Path(self.cache_dir) / (
            self.model_name.replace("/", "_") + "_compiled_model_cache.pt"
        )

        # (1.a) Load original model weights into cpu ram
        load_st = perf_counter()
        input_shape = (self.batch_size, self.im_c, self.im_h, self.im_w)
        base = CLIPVisionModel.from_pretrained(self.model_name)
        preprocessor = CLIPImageProcessorFast.from_pretrained(
            self.model_name, usefast=True
        )

        # (1.b) Compile if necessary
        # Note: tried passing artifact_bytes directly to worker,
        # this makes more sense to me but didn't formally compare
        artifact_bytes, out_msg = self.get_compile_bytes(
            base, input_shape, preprocessor, compile_cache
        )
        msg += f"{out_msg}{perf_counter() - load_st:.2E}"

        worker_cfg = _WorkerCfg(
            model_config=base.config,
            state_dict=base.state_dict(),
            preprocessor=preprocessor,
            compile_cache=compile_cache,
            device_id=0,
            input_shape=input_shape,
        )

        # (2) Pinned ring buffer init
        proc_st = perf_counter()
        buf_depth = 5 * self.concurrency  # 5 reasonable??
        buf_shape = (self.batch_size, self.im_c, self.im_h, self.im_w)
        # Persistent CPU memory buffer (TODO: better/possible to do in CUDA memory?)
        self.pinned_bufs: List[Tensor] = [
            torch.empty(buf_shape, dtype=torch.uint8, pin_memory=True)
            for _ in range(buf_depth)
        ]

        self.free_q = ctx.SimpleQueue()
        self.ready_q = ctx.SimpleQueue()
        self.out_q = ctx.SimpleQueue()

        for idx in range(buf_depth):
            self.free_q.put(idx)

        # (3) Start processes
        self.procs = [
            ctx.Process(
                target=_worker_loop,
                args=(
                    i,
                    self.pinned_bufs,
                    self.free_q,
                    self.ready_q,
                    self.out_q,
                    worker_cfg,
                ),
                daemon=False,
            )
            for i in range(self.concurrency)
        ]
        msg += f"\n\ttime to init buffers: {perf_counter() - proc_st:.2E}"
        st = perf_counter()
        for p in self.procs:
            p.start()
        msg += f"\n\ttime to startup workers: {perf_counter() - st:.2E}"
        print(msg)

    def _read_images(self, paths: Sequence[os.PathLike]) -> Tensor:
        """
        Creates a batch *th.Tensor
        """

        def _load(p: os.PathLike) -> Tensor:
            return read_image(str(vol_mnt / p))

        with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as pool:
            images = list(pool.map(_load, paths))
        return torch.stack(images)

    @modal.method()
    async def embed(self, batch_paths: List[os.PathLike]) -> Tuple[float, int]:
        """
        Encode images and return latency + count.
        TODO: add/save embeddings to FAISS vector DB would be sweet.
        """
        with torch.no_grad():
            etime = perf_counter()
            slot = self.free_q.get()  # blocking until a slot is free
            buf = self.pinned_bufs[slot]

            # ── disk → host (CPU) ──────────────────────────────────────────
            buf.copy_(self._read_images(batch_paths))
            self.ready_q.put((slot, perf_counter()))  # hand off to worker

            # ── wait for worker result ─────────────────────────────────────
            embedding, msg, inf_time = self.out_q.get()

            if self.verbose_inference:
                print(msg)
        # This is somehow significantly faster than the others but OVERALL slower (???)
        # And here are are even penalizing this method for batch creation time.
        total_embed_time = perf_counter() - etime
        return total_embed_time, len(batch_paths)  # inf_time

    @modal.exit()
    def _shutdown(self) -> None:
        """
        Gracefully terminate workers. Doesn't work oftentimes...
        """
        for _ in self.procs:
            self.ready_q.put(None)  # poison pill
        for p in self.procs:
            p.join(timeout=3)


# ## Backbone


@app.local_entrypoint()
def main():
    im_cap = 10000
    million_image_test = False  # overrides im_cap!

    gpu = "A10G"
    max_containers = 1  # NOTE: this is ignored if buffer_containers is not None

    hf_dataset_name = "microsoft/cats_vs_dogs"
    model_name: str = "openai/clip-vit-base-patch16"
    batch_size: int = 500
    im_c: int = 3
    im_h: int = 224
    im_w: int = 224
    threads_per_core: int = 4
    verbose_inference: bool = True

    allow_concurrent_inputs = 2
    n_engines: int = 2

    start_time = perf_counter()

    datadir = Path("extracted") / hf_dataset_name
    im_path_list = [
        x.path
        for x in data_volume.listdir(datadir.as_posix())
        if x.path.endswith(".jpg")
    ]

    # Dataset extension or pruning
    if million_image_test:
        print("WARNING: `million_image_test` FLAG RECEIVED!")
        mil = int(1e6)
        while len(im_path_list) < mil:
            im_path_list += im_path_list
        im_path_list = im_path_list[:mil]
    elif len(im_path_list) > im_cap:
        im_path_list = im_path_list[:im_cap]

    n_ims = len(im_path_list)

    app_cfg = {} if buffer_containers else {"max_containers": max_containers}

    embedder = MPEngine.with_options(
        gpu=gpu,
        allow_concurrent_inputs=allow_concurrent_inputs,
        **app_cfg,
    )(
        model_name=model_name,
        batch_size=batch_size,
        n_engines=n_engines,
        concurrency=allow_concurrent_inputs,
        im_c=im_c,
        im_h=im_h,
        im_w=im_w,
        threads_per_core=threads_per_core,
        verbose_inference=verbose_inference,
    )

    times, batchsizes = [], []
    for time, batchsize in embedder.embed.map(chunked(im_path_list, batch_size)):
        times.append(time)
        batchsizes.append(batchsize)

    if n_ims > 0:
        total_duration = perf_counter() - start_time
        total_throughput = n_ims / total_duration
        embed_throughputs = [
            batchsize / time for batchsize, time in zip(batchsizes, times)
        ]
        avg_throughput = sum(embed_throughputs) / len(embed_throughputs)

        log_msg = (
            f"{embedder.name}{gpu}::batch_size={batch_size}::"
            f"n_ims={n_ims}::concurrency={allow_concurrent_inputs}\n"
            f"\tTotal time:\t{total_duration / 60:.2f} min\n"
            f"\tOverall throughput:\t{total_throughput:.2f} im/s\n"
            f"\tSingle-model throughput (avg):\t{avg_throughput:.2f} im/s\n"
        )

        print(log_msg)
