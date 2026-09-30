"""Logging and token preparation for the SWE-Gym training example."""

import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tinker import types

from .environments import InfrastructureError
from .rollouts import load_episode


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


class Log:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def event(self, kind, **fields):
        row = {"time": time.time(), "event": kind, **fields}
        with self.lock:
            with (self.root / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if kind == "step_done":
                print(json.dumps(row, allow_nan=False), flush=True)
        return row


def advantages(rewards):
    a = np.asarray(rewards, dtype=np.float64)
    return (a - a.mean()) / (a.std(ddof=1) + 1e-6)


def datum(episode, advantage, token_denominator, reference_logprobs, kl_coef):
    tokens, mask, old = episode["tokens"], episode["mask"], episode["behavior_logprobs"]
    if len(reference_logprobs) != len(tokens):
        raise InfrastructureError("Reference logprobs length mismatch")
    adv = []
    for i in range(1, len(tokens)):
        if mask[i]:
            ref = reference_logprobs[i]
            if ref is None or not math.isfinite(ref):
                raise InfrastructureError(
                    "Reference logprob missing on generated token"
                )
            adv.append(
                (float(advantage) - kl_coef * (old[i] - ref)) / token_denominator
            )
        else:
            adv.append(0.0)

    def tensor(data, dtype):
        return types.TensorData(data=data, dtype=dtype, shape=[len(data)])

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tensor(tokens[1:], "int64"),
            "logprobs": tensor(old[1:], "float32"),
            "advantages": tensor(adv, "float32"),
        },
    )


def prepare_minibatch(minibatch, reference, kl_coef):
    pairs = [
        (episode, advantage)
        for group in minibatch
        for episode, advantage in zip(
            group["episodes"],
            advantages([episode["reward"] for episode in group["episodes"]]),
        )
    ]
    denominator = sum(episode["generated_tokens"] for episode, _ in pairs)
    with ThreadPoolExecutor(16) as ref_pool:

        def build(pair):
            saved, advantage = pair
            episode = load_episode(saved["path"])
            refs = reference.compute_logprobs(
                types.ModelInput.from_ints(episode["tokens"])
            ).result(timeout=3600)
            return datum(
                episode,
                advantage,
                denominator,
                refs,
                kl_coef,
            )

        data = list(ref_pool.map(build, pairs))
    return data
