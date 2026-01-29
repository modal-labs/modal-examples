# ---
# mypy: ignore-errors
# ---

# # Override Modal resource options (GPU, scaling) at runtime with `Cls.with_options`

# [`Cls.with_options`](https://modal.com/docs/reference/modal.Cls#with_options)
# lets you override the resource configuration of a
# Modal [Cls](https://modal.com/docs/guide/lifecycle-functions) at runtime.
# This is useful when the same code needs to run
# with different resource allocations -- say, with a GPU or with out,
# or with a large [warm pool of containers](https://modal.com/docs/guide/cold-start)
# -- at different times -- say, when iterating on code and when in production.

# Each call to `with_options` returns a new class handle that scales
# independently from the original.

# ## Setup

import modal

app = modal.App("example-cls-with-options")


# ## Defining the class

# We define a simple class with a method that performs a
# CPU-bound computation. The class is configured with modest defaults.


@app.cls(cpu=1, memory=128, timeout=60)
class Worker:
    @modal.method()
    def compute(self, n: int) -> int:
        import subprocess

        # if GPU available, prints details
        subprocess.Popen("nvidia-smi", shell=True)

        return sum(i * i for i in range(n))


# ## Using `with_options` to override configuration

# We can call `with_options` on the class to get a new handle
# with different resource settings.


@app.local_entrypoint()
def main():
    # Use the default configuration for a light workload
    default_worker = Worker()
    result = default_worker.compute.remote(1_000)
    print(f"Default worker result: {result}")

    # Create a GPU-accelerated variant
    GpuWorker = Worker.with_options(gpu="T4", memory=512)
    gpu_worker = GpuWorker()
    result = gpu_worker.compute.remote(10_000_000)
    print(f"GPU worker result:     {result}")
