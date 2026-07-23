# ---
# cmd: ["modal", "run", "--detach", "09_job_queues/scatter_gather.py"]
# ---

# # Scatter-gather: split a job, run the parts in parallel, gather the results

# The scatter-gather pattern splits a job into independent parts, runs those
# parts in parallel, then combines their outputs once they've all finished.

# Imagine an ML inference workload that, per input, requires two separate inference
# calls, waits for both, and stores the results only if both return. The simplest way to write
# this is a parent Function that calls each child with
# [`.remote()`](https://modal.com/docs/reference/modal.Function#remote) and blocks
# on the results. But a blocked parent is a container sitting idle and you pay for it
# to do nothing but wait. Across a large batch of jobs that's a lot of CPUs
# simply waiting for their child function calls to finish.

# The scatter-gather pattern avoids this. The parent
# [`spawn`](https://modal.com/docs/reference/modal.Function#spawn)s its parts and
# returns right away, so no blocking. Each part records where its result lives, and
# the last part to finish triggers a separate gather step. Modal auto-scales
# containers to match the work, and a Modal
# [Dict](https://modal.com/docs/guide/dicts-and-queues) lets them coordinate as
# they go, so no container is ever billed just to wait on another.

# To make it concrete, we simulate an ML inference job split across a fixed pool
# of workers. Each input is broken into parts, each part runs on its own worker,
# and whichever worker finishes last triggers the gather step that collects every
# part's output.

import time
import uuid
from contextlib import contextmanager

import modal

app = modal.App("example-scatter-gather", image=modal.Image.debian_slim())

# ## Coordinate workers with a Modal Dict

# A Modal Dict is a distributed key-value store our Functions can share.
# We use one to count how many parts of each job have finished and to hold onto
# each part's Function Call id.

state = modal.Dict.from_name("example-scatter-gather-state", create_if_missing=True)

# Multiple workers race to update the same counter, so we guard it with a simple
# distributed lock: a `put` with `skip_if_exists` succeeds for exactly one caller
# at a time, and everyone else spins until the key is freed.


@contextmanager
def lock(key):
    while not state.put(key, True, skip_if_exists=True):
        time.sleep(1)

    try:
        yield
    finally:
        state.pop(key)


# ## Scatter: fan a job out into parts

# Each job gets a unique id and is split into `n_parts` pieces. We `spawn` a
# worker for each piece and return the id right away, so the caller doesn't wait.

n_parts = 2


@app.function()
def inference(x):
    inference_id = uuid.uuid4().hex
    for part in range(n_parts):
        inference_part.spawn(x, inference_id, part)
    return inference_id


# The workers run in parallel across many containers. We cap the pool at 50 to
# simulate limits on resources.

# Each worker stashes its own Function Call id in the Dict so its result can be
# retrieved later, does its work, then increments the shared "parts done" counter
# under the lock. The worker that finishes last -- the one that pushes
# `parts_done` up to `n_parts` -- kicks off the gather step.


@app.function(max_containers=50)
def inference_part(x, inference_id, part):
    import random

    state[f"{inference_id}:{part}"] = modal.current_function_call_id()

    time.sleep(random.randint(1, 5))  # stand-in for real work
    result = f"part-{part} of {x!r}"

    with lock(f"{inference_id}:lock"):
        parts_done = state.get(f"{inference_id}:parts_done", 0) + 1
        state[f"{inference_id}:parts_done"] = parts_done

    if parts_done == n_parts:
        gather.spawn(inference_id)

    return result


# ## Gather: collect the parts once they've all finished

# We look up each part's Function Call by the id we stashed in the Dict and `get`
# its result. With every part in hand, we can assemble the final output.
# While we simply print the results here, further processing and storing
# them in a Modal volume is usually what comes here.


@app.function()
def gather(inference_id):
    call_ids = [state[f"{inference_id}:{part}"] for part in range(n_parts)]
    results = [modal.FunctionCall.from_id(call_id).get() for call_id in call_ids]
    print(f"{inference_id}: {results}")
    # persist the gathered results here


# ## Run it

# `spawn_map` invokes functions on respective inputs. Modal auto-scales workers
# to match, then scales back down to zero when the work runs out.

# ```shell
# modal run scatter_gather.py
# ```


@app.local_entrypoint()
def main():
    inference.spawn_map([f"inference data {i}" for i in range(100)])
