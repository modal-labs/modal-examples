# ---
# cmd: ["modal", "run", "03_scaling_out/multistage_pipeline.py"]
# deploy: true
# mypy: ignore-errors
# ---

# # Chain Functions into a multi-stage pipeline (with a live progress dashboard)

# Real pipelines are *chains* of Functions: load some data, transform it on CPU,
# run it through a model on GPU, write the results somewhere, then post-process.
# The stages have wildly different speeds and costs, and you want them to run
# **detached** -- you kick off the work and walk away, while the inputs flow
# through on their own.

# This example shows the Modal best practices for that pattern:

# - **Fan out** the first stage with
#   [`spawn_map`](https://modal.com/docs/reference/modal.Function#spawn_map), then
#   have each stage hand off to the next with
#   [`spawn`](https://modal.com/docs/reference/modal.Function#spawn). The whole
#   pipeline runs serverlessly after your launcher exits.
# - **Cap the expensive stage** with
#   [`max_containers`](https://modal.com/docs/guide/scale) so a fast upstream
#   stage can't stampede a slow, pricey downstream one.
# - **Track every input** -- which stage it's in, and the FunctionCall and task
#   IDs of every attempt (including retries) -- in a
#   [`modal.Dict`](https://modal.com/docs/reference/modal.Dict).
# - **Watch it live** from a tiny self-refreshing web dashboard.

# The stage bodies here are just `sleep`s so you can focus on the wiring. To build
# a real pipeline, swap in your own `process` functions in the `STAGES` table near
# the bottom -- the engine above it never changes.

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import modal

app = modal.App("example-multistage-pipeline")

MINUTES = 60  # seconds

# ## Sizing the bottleneck with `max_containers`

# Our middle stage is **1000× slower** than the first (10s vs. 0.01s). If we let
# the first stage fan out freely, it finishes almost instantly and dumps the
# entire batch onto the slow stage at once. That's actually fine -- Modal durably
# *queues* the inputs, so nothing is lost. What's *not* fine is letting the slow
# stage autoscale without bound: every container it spins up is a (GPU) machine
# you pay for.

# So we cap the bottleneck. The math is simple. If a stage takes `T` seconds per
# input, one container clears `1/T` inputs per second, and `K` containers clear
# `K/T`. To push `N` inputs through in a target wall-clock `W`:

# ```
# max_containers ≈ N * T / W
# ```

# For 100 inputs at 10s each in ~60s, that's `100 * 10 / 60 ≈ 17` containers. Pick
# the cap from your throughput target and your budget, not from how fast the
# upstream stage happens to be. Cheap, fast stages can keep a low cap (a couple of
# containers drain the batch in no time); the slow, expensive stage is the one
# worth tuning.

# A second best practice hides in here: **don't shuttle large intermediates
# through `spawn` payloads.** If the inference stage produces something sizable,
# write it to a [Volume](https://modal.com/docs/guide/volumes) and pass only a key
# or path to the next stage. We do exactly that in the final stage below.


# ## Tracking inputs as they flow

# `spawn_map` returns `None` -- it's fire-and-forget -- so we can't collect
# FunctionCall IDs from the launcher. Instead, **each stage reports its own IDs**
# the moment it starts running, using
# [`current_function_call_id()`](https://modal.com/docs/reference/modal.current_function_call_id)
# and the `MODAL_TASK_ID` environment variable. This is strictly better than
# collecting them up front: if a stage is preempted and Modal retries it in a
# fresh container, the retry reports a *new* attempt, so we capture the full
# history rather than a stale handle.

# An `Attempt` is one execution of one stage for one input.


@dataclass
class Attempt:
    stage: str
    function_call_id: str  # from current_function_call_id()
    task_id: str  # the container, from $MODAL_TASK_ID
    started_at: float
    finished_at: Optional[float] = None


# An `Item` is a single input's journey through the whole pipeline. `attempts` is
# a **list** precisely so a preempted-and-retried stage appends rather than
# overwrites -- this is the "list of IDs" that survives retries.


@dataclass
class Item:
    item_id: int
    status: str = "queued"  # "queued" | "running" | "done"
    attempts: list[Attempt] = field(default_factory=list)


# A `Run` is one launch of the pipeline over a batch of inputs. It holds the
# run-level metadata; the per-input records live under their own keys (see below).


@dataclass
class Run:
    run_id: str
    stages: list[str]
    item_ids: list[int]
    started_at: float


# The `Tracker` is our entire state layer: a `modal.Dict` shared by every stage
# and the dashboard. The one design choice that matters: we key state **per
# input** (`<run>/item/<id>`), never as one big blob. Each input is at exactly one
# stage at a time, so each key has at most one writer at a time -- no lock, no
# lost updates, no contention. (A production system might reach for a real
# database; a Dict is the batteries-included choice that keeps this example to a
# single file.)


class Tracker:
    def __init__(self, name: str):
        self.dict = modal.Dict.from_name(name, create_if_missing=True)

    @staticmethod
    def _run_key(run_id: str) -> str:
        return f"{run_id}/run"

    @staticmethod
    def _item_key(run_id: str, item_id: int) -> str:
        return f"{run_id}/item/{item_id}"

    def create_run(self, run: Run) -> None:
        for item_id in run.item_ids:
            self.dict[self._item_key(run.run_id, item_id)] = Item(item_id=item_id)
        self.dict[self._run_key(run.run_id)] = run  # write last: signals "ready"

    def start_attempt(self, run_id: str, item_id: int, attempt: Attempt) -> None:
        key = self._item_key(run_id, item_id)
        item = self.dict[key]
        item.status = "running"
        item.attempts.append(attempt)
        self.dict[key] = item

    def finish_attempt(self, run_id: str, item_id: int, *, is_last: bool) -> None:
        key = self._item_key(run_id, item_id)
        item = self.dict[key]
        item.attempts[-1].finished_at = time.time()
        if is_last:
            item.status = "done"
        self.dict[key] = item

    def get_items(self, run: Run) -> list[Item]:
        return [self.dict[self._item_key(run.run_id, i)] for i in run.item_ids]

    def latest_run(self) -> Optional[Run]:
        runs = [self.dict[k] for k in self.dict.keys() if k.endswith("/run")]
        return max(runs, key=lambda r: r.started_at, default=None)


TRACKER = Tracker("example-multistage-pipeline-state")


# ## The plug-and-play engine

# A `Stage` is your logic (`process`) plus the Modal resources it should run with.
# The `options` dict is forwarded verbatim to
# [`with_options`](https://modal.com/docs/reference/modal.Function#with_options),
# so you get `gpu`, `cpu`, `volumes`, `retries`, `max_containers`, and friends for
# free.


@dataclass(frozen=True)
class Stage:
    name: str
    process: Callable[[Any], Any]
    options: dict = field(default_factory=dict)


# Every stage runs the *same* generic `worker` Function. We then call
# `with_options` once per stage, which hands back a Function handle backed by its
# **own, independently-autoscaling container pool**. That's the elegant bit: one
# definition, N pools, each tuned separately -- a GPU pool capped at 4 here, a CPU
# pool capped at 2 there -- with no duplicated function bodies.

# The envelope passed between stages is deliberately tiny: which stage we're in,
# which run and input, and a small payload.


@app.function()
def worker(envelope: dict) -> None:
    run_id, item_id, stage_index = (
        envelope["run_id"],
        envelope["item_id"],
        envelope["stage_index"],
    )
    stage = STAGES[stage_index]
    is_last = stage_index == len(STAGES) - 1

    # Report who we are *before* doing the work, so the dashboard sees us arrive
    # and so a retry shows up as a distinct attempt.
    TRACKER.start_attempt(
        run_id,
        item_id,
        Attempt(
            stage=stage.name,
            function_call_id=modal.current_function_call_id(),
            task_id=os.environ.get("MODAL_TASK_ID", "local"),
            started_at=time.time(),
        ),
    )

    result = stage.process(envelope["payload"])  # <- your logic runs here

    TRACKER.finish_attempt(run_id, item_id, is_last=is_last)

    # Hand off to the next stage with `spawn`. The chain continues serverlessly.
    if not is_last:
        STAGE_HANDLES[stage_index + 1].spawn(
            {**envelope, "stage_index": stage_index + 1, "payload": result}
        )


def launch(num_items: int) -> Run:
    """Create a Run and fan its inputs into the first stage with `spawn_map`."""
    run = Run(
        run_id=uuid.uuid4().hex[:8],
        stages=[stage.name for stage in STAGES],
        item_ids=list(range(num_items)),
        started_at=time.time(),
    )
    TRACKER.create_run(run)
    STAGE_HANDLES[0].spawn_map(
        [
            {"run_id": run.run_id, "item_id": i, "stage_index": 0, "payload": i}
            for i in run.item_ids
        ]
    )
    return run


# ## Your pipeline, as data

# This is the only part you edit to build a real pipeline. Each `process` is an
# ordinary Python function: it takes the previous stage's output and returns the
# next stage's input.

# Stage 1 is fast and cheap, so a low `max_containers` drains the batch quickly.
# Stage 2 is the slow bottleneck -- this is where you'd attach a `gpu=...` and
# size `max_containers` to your budget. Stage 3 mounts a Volume and writes the
# final artifact.

OUTPUTS = modal.Volume.from_name(
    "example-multistage-pipeline-outputs", create_if_missing=True
)
OUTPUT_DIR = "/outputs"


def load(payload: int) -> int:
    """Stage 1: cheap CPU preprocessing (e.g. read + tokenize one dataset row)."""
    time.sleep(0.01)
    return payload


def infer(payload: int) -> int:
    """Stage 2: the slow, expensive step (e.g. batched GPU inference)."""
    time.sleep(10)
    # In a real pipeline a sizable output would be written to a Volume here, and
    # we'd return only its key -- never shuttle big blobs through `spawn`.
    return payload * payload


def write(payload: int) -> int:
    """Stage 3: persist results to a Volume and post-process."""
    time.sleep(2)
    from pathlib import Path

    (Path(OUTPUT_DIR) / f"result-{payload}.txt").write_text(str(payload))
    OUTPUTS.commit()  # make the write durable
    return payload


STAGES: list[Stage] = [
    Stage("load", load, {"cpu": 1, "max_containers": 2}),
    # The bottleneck. Add `"gpu": "L4"` for real inference; tune `max_containers`
    # to N * 10s / target_seconds (see the sizing note above).
    Stage("infer", infer, {"cpu": 1, "max_containers": 4}),
    Stage("write", write, {"max_containers": 4, "volumes": {OUTPUT_DIR: OUTPUTS}}),
]

# Specialize the one `worker` into one independently-scaling pool per stage.
STAGE_HANDLES = [worker.with_options(**stage.options) for stage in STAGES]


# ## A snapshot of the whole pipeline

# One pure function turns the raw records into everything we want to show:
# where each input is, how many are in flight per stage, and live throughput. The
# terminal launcher and the web dashboard both render from this single source of
# truth.


def snapshot(run: Run, items: list[Item]) -> dict:
    elapsed = max(time.time() - run.started_at, 1e-9)
    stats = {
        name: {"in_progress": 0, "finished": 0, "latency": 0.0} for name in run.stages
    }
    locations, done = [], 0

    for item in items:
        last = item.attempts[-1] if item.attempts else None
        if item.status == "done":
            done += 1
            location = "✓ done"
        elif last is None:
            location = "· queued"
        elif last.finished_at is None:
            stats[last.stage]["in_progress"] += 1
            location = f"▶ {last.stage}"
        else:  # finished a stage, waiting to be picked up by the next one
            nxt = run.stages[run.stages.index(last.stage) + 1]
            location = f"⏳ → {nxt}"
        # A clean input has one attempt per stage; anything extra is a retry.
        retries = len(item.attempts) - len({a.stage for a in item.attempts})
        locations.append((item.item_id, location, retries))

    for item in items:
        for attempt in item.attempts:
            if attempt.finished_at is not None:
                s = stats[attempt.stage]
                s["finished"] += 1
                s["latency"] += attempt.finished_at - attempt.started_at
    for s in stats.values():
        s["latency"] = s["latency"] / s["finished"] if s["finished"] else 0.0

    return {
        "elapsed": elapsed,
        "done": done,
        "total": len(items),
        "throughput": done / elapsed,
        "stages": stats,
        "locations": locations,
    }


# ## The live dashboard

# A dependency-free, server-rendered HTML page that refreshes itself once a
# second -- the simplest possible "watch the pipeline" view. Deploy or
# `modal serve` this file and open the printed URL.


web_image = modal.Image.debian_slim().uv_pip_install("fastapi[standard]==0.116.0")


@app.function(image=web_image)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def dashboard():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    web_app = FastAPI()

    @web_app.get("/", response_class=HTMLResponse)
    def index():
        run = TRACKER.latest_run()
        if run is None:
            return "<h2>No pipeline runs yet.</h2>"
        return render(run, snapshot(run, TRACKER.get_items(run)))

    return web_app


def render(run: Run, snap: dict) -> str:
    cards = "".join(
        f"""<div class=card>
              <h3>{name}</h3>
              <div class=big>{s["in_progress"]}</div><div class=sub>in flight</div>
              <div>{s["finished"]} done · {s["latency"]:.2f}s avg</div>
            </div>"""
        for name, s in snap["stages"].items()
    )
    rows = "".join(
        f"<tr><td>{i}</td><td>{loc}</td><td>{('⟳ ' + str(n)) if n else '—'}</td></tr>"
        for i, loc, n in snap["locations"]
    )
    return f"""<!doctype html><html><head>
      <meta http-equiv=refresh content=1>
      <title>pipeline {run.run_id}</title>
      <style>
        body{{font:14px ui-monospace,monospace;background:#0b0f14;color:#d7e0ea;margin:2rem}}
        .cards{{display:flex;gap:1rem;margin:1rem 0}}
        .card{{background:#141b24;border:1px solid #233;border-radius:10px;padding:1rem;min-width:120px}}
        .big{{font-size:2rem;font-weight:700;color:#5fd}}.sub{{color:#789;margin-bottom:.5rem}}
        h3{{margin:0 0 .5rem;color:#9cf}}table{{border-collapse:collapse;width:100%}}
        td,th{{text-align:left;padding:.3rem .8rem;border-bottom:1px solid #1c252f}}
      </style></head><body>
      <h1>pipeline run <code>{run.run_id}</code></h1>
      <p>{snap["done"]}/{snap["total"]} done ·
         <b>{snap["throughput"]:.2f} items/s</b> · {snap["elapsed"]:.0f}s elapsed</p>
      <div class=cards>{cards}</div>
      <h2>where is each input?</h2>
      <table><tr><th>input</th><th>location</th><th>retries</th></tr>{rows}</table>
    </body></html>"""


# ## Launch it

# Running `modal run` fans out the inputs and then tails progress in your terminal
# until everything finishes (which keeps this ephemeral run alive long enough for
# the detached chain to complete). In production you'd `modal deploy` instead and
# trigger `launch` from anywhere, letting the pipeline run fully detached.


@app.local_entrypoint()
def main(num_items: int = 8):
    run = launch(num_items)
    print(f"launched run {run.run_id} with {num_items} inputs")
    print(f"watch live:  modal serve {__file__}  (then open the dashboard URL)\n")

    deadline = time.time() + 15 * MINUTES
    while time.time() < deadline:
        snap = snapshot(run, TRACKER.get_items(run))
        flight = "  ".join(
            f"{name}:{s['in_progress']}▶/{s['finished']}✓"
            for name, s in snap["stages"].items()
        )
        print(
            f"[{snap['elapsed']:6.1f}s] {snap['done']}/{snap['total']} done "
            f"· {snap['throughput']:.2f}/s  ||  {flight}"
        )
        if snap["done"] == snap["total"]:
            print("\npipeline complete 🎉")
            break
        time.sleep(2)
