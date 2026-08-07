# # Orchestrate a multi-step pipeline with Modal Functions

# Every step of this pipeline is a Modal Function that hands off to the next one,
# so you can run each stage with its own resources without standing up a separate orchestrator. A step looks up its successor by name
# with [`Function.from_name`](https://modal.com/docs/guide/trigger-deployed-functions)
# and starts it with [`Function.spawn`](https://modal.com/docs/guide/trigger-deployed-functions#invocation-patterns).

# The toy computation used here is to build a range of numbers, square them, and sum them. Each
# step caches its output on a Volume, so a rerun skips work already done.

# Run it and see its trace:

# ```bash
# modal run 09_job_queues/pipeline_orchestration.py --n 10
# ```

# Because the steps hand off by name against a pinned version, the App has to be
# deployed before it runs. The entrypoint deploys it for you if it isn't already,
# but you can also deploy explicitly:

# ```bash
# modal deploy 09_job_queues/pipeline_orchestration.py
# ```

# Run that again with the same `n` and every step hits the cache. Deploy again and
# every step recomputes regardless of which step you edited because keys are constructed partly by App version,
# which a deploy bumps for the whole App.

# ## Set up

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
MINUTES = 60  # seconds

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.2.6")

with image.imports():
    import numpy as np

# A [Dict](https://modal.com/docs/guide/dicts-and-queues) holds each run's trace and a
# [Volume](https://modal.com/docs/guide/volumes) holds the input and the artifacts
# steps pass along.

state = modal.Dict.from_name(f"{APP_NAME}-state", create_if_missing=True)
data = modal.Volume.from_name(f"{APP_NAME}-data", create_if_missing=True)


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
    input_id: str  # identifies the input and names its directory on the Volume
    function_call_ids: list[str] = field(default_factory=list)  # `fc-`, per step
    done: bool = False


# ## Construct artifact keys

# A step skips work whose output is already on the Volume, so its key changes
# whenever its output would. A key is constructed by the deployed App version
# and the step's inputs, which enter as its predecessor's key. Chaining
# the keys means both only have to enter once, at the seed.


def cache_key(pipeline: Pipeline, step_num: int) -> str:
    key = f"{pipeline.input_id}@v{pipeline.app_version}"  # the version seeds the chain
    for step in STEPS[: step_num + 1]:
        digest = hashlib.sha256(f"{step.name}/{key}".encode()).hexdigest()
        key = f"{step.name}-{digest[:16]}"
    return key


def artifact(pipeline: Pipeline, step_num: int) -> Path:
    return DATA_DIR / cache_key(pipeline, step_num) / STEPS[step_num].output


# ## Coordinate work handoff

# Whoever spawns a step stamps its call id onto the run, so the trace builds up as the
# pipeline moves and always ends with the step that is currently pending. Every
# hand-off goes through the one pinned lookup in `step_function`.


def start_step(pipeline: Pipeline, step_num: int) -> Path:
    data.reload()  # a container only sees the Volume as of when it started
    pipeline.function_call_ids.append(modal.current_function_call_id())

    out = artifact(pipeline, step_num)
    out.parent.mkdir(parents=True, exist_ok=True)
    status = "cached" if out.exists() else "computing"
    print(f"[{STEPS[step_num].name}] {status} {out.name}")
    return out


def step_function(step_num: int, app_version: int) -> modal.Function:
    return modal.Function.from_name(APP_NAME, STEPS[step_num].name, version=app_version)


def spawn_step(pipeline: Pipeline, step_num: int) -> modal.FunctionCall:
    step = step_function(step_num, pipeline.app_version)
    call = step.spawn(pipeline, step_num)
    pipeline.function_call_ids.append(call.object_id)
    state[pipeline.run_id] = pipeline
    return call


def spawn_next(pipeline: Pipeline, step_num: int) -> None:
    if step_num + 1 >= len(STEPS):
        pipeline.done = True
        state[pipeline.run_id] = pipeline
        print("[done]")
        return

    spawn_step(pipeline, step_num + 1)


# ## Save the results

# The data for each step must be persisted to the Volume before the next step starts.
# [Background commits](https://modal.com/docs/guide/volumes#background-commits) also
# land every few seconds, but on their own schedule, so we run `Volume.commit()`
# manually before each hand-off. Committing doesn't make the write atomic, so a crash
# mid-write can still leave a corrupted artifact.


@app.function(image=image, volumes={DATA_DIR: data})
def build(pipeline: Pipeline, step_num: int) -> None:
    out = start_step(pipeline, step_num)
    if not out.exists():
        with open(DATA_DIR / pipeline.input_id / "input.json") as f:
            n = json.load(f)["n"]
        np.save(out, np.arange(1, n + 1))
        data.commit()
    spawn_next(pipeline, step_num)


@app.function(image=image, volumes={DATA_DIR: data})
def square(pipeline: Pipeline, step_num: int) -> None:
    out = start_step(pipeline, step_num)
    if not out.exists():
        numbers = np.load(artifact(pipeline, step_num - 1))
        np.save(out, numbers**2)
        data.commit()
    spawn_next(pipeline, step_num)


@app.function(image=image, volumes={DATA_DIR: data})
def total(pipeline: Pipeline, step_num: int) -> None:
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

# The driver reads the version that's live now, deploys if there isn't one, and pins
# the run to it. Note, version pinning is a
# [Team and Enterprise feature](https://modal.com/docs/guide/trigger-deployed-functions#version-pinned-lookups).


def latest_version() -> int | None:
    history = subprocess.run(
        ["modal", "app", "history", APP_NAME, "--json"], capture_output=True, text=True
    )
    versions = json.loads(history.stdout) if history.returncode == 0 else []
    numbers = [str(v.get("version", "")).removeprefix("v") for v in versions]
    return max((int(n) for n in numbers if n.isdigit()), default=None)


def deployed_version() -> int | None:
    version = latest_version()
    if version is None:
        return None
    try:
        step_function(0, version).hydrate()
    except modal.exception.NotFoundError:
        return None
    return version


def ensure_deployed() -> int:
    version = deployed_version()
    if version is None:
        subprocess.run(["modal", "deploy", __file__], check=True)
        version = deployed_version()
    if version is None:
        raise RuntimeError(f"no version to pin to: modal app history {APP_NAME}")
    return version


def stage_input(n: int) -> str:
    input_id = f"n-{n}"
    try:
        staged = bool(data.listdir(f"{input_id}/input.json"))
    except (FileNotFoundError, modal.exception.NotFoundError):
        staged = False
    if not staged:
        with data.batch_upload() as batch:
            blob = io.BytesIO(json.dumps({"n": n}).encode())
            batch.put_file(blob, f"{input_id}/input.json")
    return input_id


def trigger(n: int, app_version: int) -> str:
    pipeline = Pipeline(
        run_id=f"run-{uuid.uuid4().hex[:8]}",
        app_version=app_version,
        input_id=stage_input(n),
    )
    call = spawn_step(pipeline, 0)
    print(f"Started {pipeline.run_id} on v{pipeline.app_version}: {call.object_id}")
    return pipeline.run_id


def wait(run_id: str, timeout: int = 5 * MINUTES) -> Pipeline:
    deadline = time.time() + timeout
    while time.time() < deadline:
        pipeline = state.get(run_id)
        if pipeline is not None and pipeline.function_call_ids:
            if pipeline.done:
                return pipeline
            call_id = pipeline.function_call_ids[-1]
            try:
                modal.FunctionCall.from_id(call_id).get(timeout=0)
            except TimeoutError:
                pass
        time.sleep(1)
    raise TimeoutError(f"{run_id} did not finish in {timeout}s")


# The trace prints each step's key, so a rerun of the same input on the same code
# shows the same keys.


def report(pipeline: Pipeline) -> None:
    print(f"\n{pipeline.run_id} finished on v{pipeline.app_version}:")
    for step_num, call_id in enumerate(pipeline.function_call_ids):
        print(f"  {cache_key(pipeline, step_num)}  function_call={call_id}")
    final_key = cache_key(pipeline, len(STEPS) - 1)
    result = json.loads(b"".join(data.read_file(f"{final_key}/{STEPS[-1].output}")))
    print(f"  result: sum of {result['count']} squares = {result['total']}\n")


@app.local_entrypoint()
def run(n: int = 10) -> None:
    app_version = ensure_deployed()
    report(wait(trigger(n, app_version)))
