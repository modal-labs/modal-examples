# # Orchestrate a multi-step pipeline with Modal Functions

# Every step of this pipeline is a Modal Function that hands off to the next one,
# so no external orchestrator is needed. A step looks up its successor by name
# with [`Function.from_name`](https://modal.com/docs/guide/trigger-deployed-functions)
# and starts it with `Function.spawn`.

# The toy computation is: build a range of numbers, square them, sum them. Each
# step caches its output on a Volume, so a rerun skips work already done.

# First, deploy the App:

# ```bash
# modal deploy pipeline_orchestration.py
# ```

# Then trigger a run and see its trace:

# ```bash
# modal run pipeline_orchestration.py --n 10
# ```

# Run that again with the same `n` and every step hits the cache. Edit a step and
# deploy again and they all recompute.

import hashlib
import io
import json
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import modal

APP_NAME = "example-pipeline-orchestration"
DATA_DIR = Path("/data")

app = modal.App(APP_NAME)
image = modal.Image.debian_slim().pip_install("numpy")

with image.imports():
    import numpy as np

# A [Dict](https://modal.com/docs/guide/dicts-and-queues) holds each run's trace and a
# [Volume](https://modal.com/docs/guide/volumes) holds the input and the artifacts
# steps pass along.

state = modal.Dict.from_name(f"{APP_NAME}-state", create_if_missing=True)
data = modal.Volume.from_name(f"{APP_NAME}-data", create_if_missing=True)

# Each step is a Function and its outputted artifact.


class Step(NamedTuple):
    name: str  # the Function that runs it
    output: str  # the artifact it leaves under its key


STEPS = [
    Step("build", "numbers.npy"),
    Step("square", "squared.npy"),
    Step("total", "total.json"),
]


@dataclass
class Pipeline:
    run_id: str  # unique per execution
    app_version: int  # deployed code version this run is pinned to
    input_id: str  # identifies the input, names its directory on the Volume
    function_call_ids: list[str] = field(default_factory=list)  # `fc-`, per step
    done: bool = False  # set by the last step; `wait` polls for it


# ## Key each artifact by the code and inputs that made it

# A step skips work whose output is already on the Volume, so the key naming that
# output has to move whenever the output would. Two ingredients move it:

# - The App version, which a deploy bumps whenever the code changes. Keying on the
# pipeline's input alone would serve a stale artifact after you edit a step.
# - The step's own inputs, folded in as its predecessor's key rather than as the
# pipeline's input. Two pipelines that both run `square` on different arrays would
# otherwise share a key like `n-10` and read each other's work.

# Both enter once, at the seed: every link folds in a predecessor key that already
# carries them, so a new source step would have to seed the version itself.


def cache_key(pipeline: Pipeline, step_num: int) -> str:
    """Content-address a step's output: same code and inputs, same key."""
    key = f"{pipeline.input_id}@v{pipeline.app_version}"  # the version seeds the chain
    for step in STEPS[: step_num + 1]:
        digest = hashlib.sha256(f"{step.name}/{key}".encode()).hexdigest()
        key = f"{step.name}-{digest[:16]}"
    return key


def artifact(pipeline: Pipeline, step_num: int) -> Path:
    """Where the given step keeps its output, under its key on the Volume."""
    return DATA_DIR / cache_key(pipeline, step_num) / STEPS[step_num].output


# ## Hand off from one step to the next

# Each step stamps its call id onto the run so the trace builds up as the pipeline
# moves, then hands off. Every hand-off, including the driver's, goes through the
# one pinned lookup below.


def start_step(pipeline: Pipeline, step_num: int) -> Path:
    """Record the running step on the pipeline and return its output path."""
    data.reload()  # a container only sees the Volume as of when it started
    pipeline.function_call_ids.append(modal.current_function_call_id())
    state[pipeline.run_id] = pipeline

    out = artifact(pipeline, step_num)
    out.parent.mkdir(parents=True, exist_ok=True)
    status = "cached" if out.exists() else "computing"
    print(f"[{STEPS[step_num].name}] {status} {out.name}")
    return out


def spawn_step(pipeline: Pipeline, step_num: int) -> modal.FunctionCall:
    """Start a step on the App version this run is pinned to."""
    step = modal.Function.from_name(
        APP_NAME, STEPS[step_num].name, version=pipeline.app_version
    )
    return step.spawn(pipeline, step_num)


def spawn_next(pipeline: Pipeline, step_num: int) -> None:
    """Hand off to the following step, or mark the pipeline finished."""
    if step_num + 1 >= len(STEPS):
        pipeline.done = True
        state[pipeline.run_id] = pipeline
        print("[done]")
        return

    spawn_step(pipeline, step_num + 1)


# ## Write the steps

# Every step has the same shape: start the step, do the work, spawn the next. The
# commit has to land before the hand-off, since the next step starts before this
# container shuts down and triggers Modal's automatic commit.

# Note that [background commits](https://modal.com/docs/guide/volumes#background-commits)
# land every few seconds, so a step that dies mid-write leaves a partial artifact
# that a later run would read as a hit. Production pipelines switch on a temporary
# path or a marker file so a key only appears once its artifact is whole.


@app.function(image=image, volumes={DATA_DIR: data})
def build(pipeline: Pipeline, step_num: int) -> None:
    """Materialize the input range from the input staged on the Volume."""
    out = start_step(pipeline, step_num)
    if not out.exists():
        with open(DATA_DIR / pipeline.input_id / "input.json") as f:
            n = json.load(f)["n"]
        np.save(out, np.arange(1, n + 1))
        data.commit()
    spawn_next(pipeline, step_num)


@app.function(image=image, volumes={DATA_DIR: data})
def square(pipeline: Pipeline, step_num: int) -> None:
    """Square every element of the previous step's array."""
    out = start_step(pipeline, step_num)
    if not out.exists():
        numbers = np.load(artifact(pipeline, step_num - 1))
        np.save(out, numbers**2)
        data.commit()
    spawn_next(pipeline, step_num)


@app.function(image=image, volumes={DATA_DIR: data})
def total(pipeline: Pipeline, step_num: int) -> None:
    """Reduce to the final result, small enough to hand back as JSON."""
    out = start_step(pipeline, step_num)
    if not out.exists():
        # Reaches back past `square` to also read what `build` wrote, by key.
        squared = np.load(artifact(pipeline, step_num - 1))
        numbers = np.load(artifact(pipeline, 0))
        with open(out, "w") as f:
            json.dump({"total": int(squared.sum()), "count": numbers.size}, f)
        data.commit()
    spawn_next(pipeline, step_num)


# ## Trigger a run from a local driver

# The driver reads the app version off the CLI and hands it to the run.
# Pinning lookups to a version is a [Team and Enterprise feature](https://modal.com/docs/guide/trigger-deployed-functions#version-pinned-lookups).


def deployed_version() -> int:
    """Read the App's live version, the code version this run pins to."""
    history = subprocess.run(
        ["modal", "app", "history", APP_NAME, "--json"], capture_output=True, text=True
    )
    if history.returncode != 0:
        raise RuntimeError(f"deploy the app first: modal deploy {Path(__file__).name}")
    versions = json.loads(history.stdout)
    return max(int(v["version"].removeprefix("v")) for v in versions)


def stage_input(n: int) -> str:
    """Put the input on the Volume under an id, unless it's already staged."""
    input_id = f"n-{n}"
    try:
        data.listdir(f"{input_id}/input.json")
    except modal.exception.NotFoundError:
        with data.batch_upload() as batch:
            blob = io.BytesIO(json.dumps({"n": n}).encode())
            batch.put_file(blob, f"{input_id}/input.json")
    return input_id


def trigger(n: int) -> str:
    """Stage the input, then start the first step on a pinned App version."""
    pipeline = Pipeline(
        run_id=f"run-{uuid.uuid4().hex[:12]}",
        app_version=deployed_version(),
        input_id=stage_input(n),
    )
    state[pipeline.run_id] = pipeline

    call = spawn_step(pipeline, 0)
    print(f"Started {pipeline.run_id} on v{pipeline.app_version}: {call.object_id}")
    return pipeline.run_id


def wait(run_id: str, timeout: int = 60) -> Pipeline:
    """Poll `state` until the last step marks the run done."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pipeline = state.get(run_id)
        if pipeline is not None and pipeline.done:
            return pipeline
        time.sleep(1)
    raise TimeoutError(f"{run_id} did not finish in {timeout}s")


# The trace prints each step's key, so a rerun of the same input on the same code
# shows the same keys — those are the artifacts it reused.


def report(pipeline: Pipeline) -> None:
    """Print the run's trace, then pull just the final result off the Volume."""
    print(f"\n{pipeline.run_id} finished on v{pipeline.app_version}:")
    for step_num, call_id in enumerate(pipeline.function_call_ids):
        print(f"  {cache_key(pipeline, step_num)}  function_call={call_id}")
    final_key = cache_key(pipeline, len(STEPS) - 1)
    result = json.loads(b"".join(data.read_file(f"{final_key}/{STEPS[-1].output}")))
    print(f"  result: sum of {result['count']} squares = {result['total']}\n")


@app.local_entrypoint()
def run(n: int = 10) -> None:
    """Run the pipeline end to end against the deployed App."""
    report(wait(trigger(n)))
