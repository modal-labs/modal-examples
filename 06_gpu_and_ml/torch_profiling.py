# # Tracing and profiling GPU-accelerated PyTorch programs on Modal

# ![A PyTorch trace loaded into ui.perfetto.dev](https://modal-public-assets.s3.amazonaws.com/tmpx_2c9bl5_c5aa7ab0.webp)

# GPUs are high-performance computing devices. For high-performance computing,
# tools for measuring and investigating performance are as critical
# as tools for testing and confirming correctness in typical software.

# In this example, we demonstrate how to wrap a Modal Function with PyTorch's
# built-in profiler, which captures events on both CPUs & GPUs. We also show
# how to host TensorBoard, which includes useful visualizations and
# performance improvement suggestions.

# For a live walkthrough, check out
# [this video on our YouTube channel](https://www.youtube.com/watch?v=4cesQJLyHA8).

# ## Saving traces to a Modal Volume

# Most tracing tools, including PyTorch's profiler, produce results as files on disk.
# Modal Functions run in ephemeral containers in Modal's cloud infrastructure,
# so by default these files disappear as soon as the Function finishes running.

# We can ensure these files persist by saving them to a
# [Modal Volume](https://modal.com/docs/guide/volume).
# Volumes are a distributed file system: files can be read or written from
# by many machines across a network, in this case from inside any Modal Function.

# To start, we just create a Volume with a specific name.
# We'll also set a particular directory that we'll use for it
# in our Functions below, for convenience.


from pathlib import Path

import modal

traces = modal.Volume.from_name("example-traces", create_if_missing=True)
TRACE_DIR = Path("/traces")

# ## Setting up a Modal App with a GPU-accelerated PyTorch Function

# We next set up the Modal Function that we wish to profile.

# In general, we want to attach profiling tools to code that's already in place
# and measure or debug its performance, and then detach it as easily as possible
# so that we can be confident that the same performance characteristics pertain in production.

# In keeping with that workflow, in this example we first define the Modal Function we want to profile,
# without including any of the profiling logic.

# That starts with the Function's environment: the Modal [App](https://modal.com/docs/guide/apps)
# the Function is attached to, the container [Image](https://modal.com/docs/guide/custom-container)
# with the Function's dependencies, and the hardware requirements of the Function, like a
# [GPU](https://modal.com/docs/guide/cuda).


app = modal.App("example-torch-profiling")  # create an App

image = modal.Image.debian_slim(  # define dependencies
    python_version="3.11"
).pip_install("torch==2.5.1", "numpy==2.1.3")

with image.imports():  # set up common imports
    import torch

# Here, we define the config as a dictionary so that we can re-use it here
# and later, when we attach the profiler. We want to make sure the profiler is in the same environment!

config = {"gpu": "a10g", "image": image}

# The Function we target for profiling appears below. It's just some simple PyTorch logic
# that repeatedly multiplies a random matrix with itself.

# The logic is simple, but it demonstrates two common issues with
# GPU-accelerated Python code that are relatively easily fixed:
# 1. Slowing down the issuance of work to the GPU
# 2. Providing insufficient work for the GPU to complete

# We'll cover these in more detail once we have the profiler set up.


@app.function(**config)
def underutilize(scale=1):
    records = []

    x = torch.randn(  # 🐌 2: not enough work to keep the GPU busy
        scale * 100, scale * 100, device="cuda"
    )

    for ii in range(10):
        x = x @ x

        class Record:  # 🐌 1: heavy Python work in the hot loop
            def __init__(self, value):
                self.value = value

        records.append(Record(ii))

    x[0][0].cpu()  # force a host sync for accurate timing


# ## Wrapping a Modal Function with a profiler

# Now, let's wrap our `underutilize` Function with another Modal Function
# that runs PyTorch's profiler while executing it.

# This Function has the same environment `config` as `underutilize`,
# but it also attaches a remote Modal Volume to save profiler outputs.

# To increase the flexibility of this approach, we allow it to take the target Function's name
# as an argument. That's not much use here where there's only one Function,
# but it makes it easier to copy-paste this code into your projects to add profiling.


@app.function(volumes={TRACE_DIR: traces}, **config)
def profile(
    function,
    label: str = None,
    steps: int = 3,
    schedule=None,
    record_shapes: bool = False,
    profile_memory: bool = False,
    with_stack: bool = False,
    print_rows: int = 0,
    **kwargs,
):
    from uuid import uuid4

    if isinstance(function, str):
        try:
            function = app.registered_functions[function]
        except KeyError:
            raise ValueError(f"Function {function} not found")
    function_name = function.tag

    output_dir = (
        TRACE_DIR / (function_name + (f"_{label}" if label else "")) / str(uuid4())
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if schedule is None:
        if steps < 3:
            raise ValueError("Steps must be at least 3 when using default schedule")
        schedule = {"wait": 1, "warmup": 1, "active": steps - 2, "repeat": 0}

    schedule = torch.profiler.schedule(**schedule)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ) as prof:
        for _ in range(steps):
            function.local(**kwargs)  # <-- here we wrap the target Function
            prof.step()

    if print_rows:
        print(
            prof.key_averages().table(sort_by="cuda_time_total", row_limit=print_rows)
        )

    trace_path = sorted(
        output_dir.glob("**/*.pt.trace.json"),
        key=lambda pth: pth.stat().st_mtime,
        reverse=True,
    )[0]

    print(f"trace saved to {trace_path.relative_to(TRACE_DIR)}")

    return trace_path.read_text(), trace_path.relative_to(TRACE_DIR)


# ## Triggering profiled execution from the command line and viewing in Perfetto

# We wrap one more layer to make this executable from the command line:
# a `local_entrypoint` that runs

# ```bash
# modal run torch_profiling.py --function underutilize --print-rows 10
# ```


@app.local_entrypoint()
def main(
    function: str = "underutilize",
    label: str = None,
    steps: int = 3,
    schedule=None,
    record_shapes: bool = False,
    profile_memory: bool = False,
    with_stack: bool = False,
    print_rows: int = 10,
    kwargs_json_path: str = None,
):
    if kwargs_json_path is not None:  # use to pass arguments to function
        import json

        kwargs = json.loads(Path(kwargs_json_path).read_text())
    else:
        kwargs = {}

    results, remote_path = profile.remote(
        function,
        label=label,
        steps=steps,
        schedule=schedule,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        print_rows=print_rows,
        **kwargs,
    )

    output_path = Path("/tmp") / remote_path.name
    output_path.write_text(results)
    print(f"trace saved locally at {output_path}")


# Underneath the profile results, you'll also see the path at which the trace was saved on the Volume
# and the path at which it was saved locally.

# You can view the trace in the free online [Perfetto UI](https://ui.perfetto.dev).

# ### Improving the performance of our dummy test code

# The `underutilize` demonstrates two common patterns that leads to unnecessarily low GPU utilization:
# 1. Slowing down the issuance of work to the GPU
# 2. Providing insufficient work for the GPU to complete

# We simulated 1 in `underutilize` by defining a Python class in the middle of the matrix multiplication loop.
# This takes on the order of 10 microseconds, roughly the same time it takes our A10 GPU to do the matrix multiplication.
# Move it out of the loop to observe a small improvement in utilization. In a real setting,
# this code might be useful logging or data processing logic, which we must carefully keep
# out of the way of the code driving work on the GPU.

# We simulated 2 in `underutilize` by providing a matrix that is too small to occupy the GPU for long.
# Increase the size of the matrix by a factor of 4 in each dimension (a factor of 16 total),
# to increase the utilization without increasing the execution time.

# This is an untuitive feature of GPU programming in general: much work is done concurrently
# and bottlenecks are non-obvious, so sometimes more work can be done for free or on the cheap.
# In a server for large generative models, this might mean producing multiple outputs per user
# or handling multiple users at the same time is more economical than it at first seems!

# ## Serving TensorBoard on Modal to view PyTorch profiles and traces

# The TensorBoard experiment monitoring server also includes a plugin
# for viewing and interpreting the results of PyTorch profiler runs:
# the `torch_tb_profiler` plugin.


tb_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "tensorboard==2.18.0", "torch_tb_profiler==0.4.3"
)

# Because TensorBoard is a WSGI app, we can [host it on Modal](https://modal.com/docs/guide/webhooks)
# with the `modal.wsgi_app` decorator.

# Making this work with Modal requires one extra step:
# we add some [WSGI Middleware](https://peps.python.org/pep-3333/) that checks the Modal Volume for updates
# whenever the whole page is reloaded.


class VolumeMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        if (route := environ.get("PATH_INFO")) in ["/", "/modal-volume-reload"]:
            try:
                traces.reload()
            except Exception as e:
                print("Exception while re-loading traces: ", e)
            if route == "/modal-volume-reload":
                environ["PATH_INFO"] = "/"  # redirect
        return self.app(environ, start_response)


# You can deploy the TensorBoard server defined below with the following command:
# ```bash
# modal deploy torch_profiling
# ```

# and you can find your server at the URL printed to the terminal.


@app.function(
    volumes={TRACE_DIR: traces},
    image=tb_image,
    max_containers=1,  # single replica
    scaledown_window=5 * 60,  # five minute idle time
)
@modal.concurrent(max_inputs=100)  # 100 concurrent request threads
@modal.wsgi_app()
def tensorboard():
    import tensorboard

    board = tensorboard.program.TensorBoard()
    board.configure(logdir=str(TRACE_DIR))
    (data_provider, deprecated_multiplexer) = board._make_data_provider()
    wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
        board.flags,
        board.plugin_loaders,
        data_provider,
        board.assets_zip_provider,
        deprecated_multiplexer,
        experimental_middlewares=[VolumeMiddleware],
    )

    return wsgi_app._create_wsgi_app()
