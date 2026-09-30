"""Generate exact-token multi-turn trajectories and grade them in fresh Sandboxes."""

import gzip
import json
import math
import re
import shlex
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tinker import types

from .environments import InfrastructureError, command

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command in /testbed to inspect, edit, and test the repository.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]
SYSTEM = """You are a software engineer fixing a repository issue. Use the bash tool
to inspect and edit /testbed, and run relevant tests. The testbed conda environment
is activated for every command. All shell commands are isolated and have no network
access. Make a minimal correct source fix; do not weaken or delete tests. When the
fix is complete, respond with a short final explanation without a tool call."""


def parse_commands(text):
    text = text.split("</think>")[-1]
    calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    result = []
    for call in calls:
        match = re.fullmatch(
            r"<function=bash>\s*<parameter=command>\s*(.*?)\s*</parameter>\s*</function>",
            call,
            re.S,
        )
        if not match:
            raise ValueError("Expected bash(command) in the documented XML tool format")
        result.append(match.group(1))
    if "<tool_call>" in text and not calls:
        raise ValueError("Incomplete tool call")
    return result


def generation_budget(cfg, prompt_tokens):
    # Reference-logprob requests also need backend-reserved tokens. Keep the
    # complete sampled trajectory below that endpoint's 131066-token limit.
    return min(cfg["turn_tokens"], cfg["context_tokens"] - prompt_tokens - 16)


def generate_rollout(inst, sampler, tokenizer, environments, cfg, log, identity):
    start = time.time()
    log.event("rollout_start", **identity, instance_id=inst["instance_id"])
    path = (
        log.root
        / "episodes"
        / identity["phase"]
        / str(identity["job"])
        / f"g{identity['group']:06d}-a{identity['attempt']}"
    )
    path.mkdir(parents=True, exist_ok=True)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": inst["problem_statement"]},
    ]
    tokens = list(
        tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
    )
    behavior = [0.0] * len(tokens)
    mask = [0] * len(tokens)
    turns = []
    termination = "turn_limit"
    with environments.sandbox(inst) as sb:
        environments.prepare_agent(sb)
        for turn in range(cfg["max_turns"]):
            budget = generation_budget(cfg, len(tokens))
            if budget < 1:
                termination = "context_limit"
                break
            sample_start = time.time()
            sample = (
                sampler.sample(
                    types.ModelInput.from_ints(tokens),
                    1,
                    types.SamplingParams(
                        max_tokens=budget, temperature=cfg["temperature"], top_p=1.0
                    ),
                )
                .result(timeout=3600)
                .sequences[0]
            )
            generated, lp = list(sample.tokens), list(sample.logprobs or [])
            if (
                not generated
                or len(lp) != len(generated)
                or not all(math.isfinite(x) for x in lp)
            ):
                raise InfrastructureError(
                    "Missing/nonfinite sampling log probabilities"
                )
            sample_s = time.time() - sample_start
            text = tokenizer.decode(generated, skip_special_tokens=False)
            row = {
                "turn": turn,
                "prompt_tokens": len(tokens),
                "tokens": len(generated),
                "sample_s": sample_s,
                "text": text,
                "stop_reason": str(sample.stop_reason),
            }
            turns.append(row)
            log.event(
                "sample", **identity, **{k: v for k, v in row.items() if k != "text"}
            )
            tokens.extend(generated)
            behavior.extend(lp)
            mask.extend([1] * len(generated))
            if len(generated) >= budget:
                termination = "generation_limit"
                break
            try:
                cmds = parse_commands(text)
                feedback = []
                for cmd in cmds:
                    ts = time.time()
                    rc, output = command(
                        sb,
                        "source /opt/miniconda3/bin/activate testbed && cd /testbed && "
                        + "bash -c "
                        + shlex.quote(cmd),
                        120,
                    )
                    feedback.append(f"exit_code={rc}\n{output[-20000:]}")
                    log.event(
                        "tool",
                        **identity,
                        turn=turn,
                        elapsed_s=time.time() - ts,
                        exit_code=rc,
                    )
                if not cmds:
                    termination = "final"
                    break
            except ValueError as exc:
                feedback = [str(exc)]
            suffix = (
                ""
                if tokens[-1] == tokenizer.convert_tokens_to_ids("<|im_end|>")
                else "<|im_end|>"
            )
            suffix += "\n<|im_start|>user"
            for output in feedback:
                suffix += "\n<tool_response>\n" + output.strip() + "\n</tool_response>"
            suffix += "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            obs = tokenizer.encode(suffix, add_special_tokens=False)
            tokens.extend(obs)
            behavior.extend([0.0] * len(obs))
            mask.extend([0] * len(obs))
        with gzip.open(path / "ungraded.json.gz", "wt") as f:
            json.dump(
                {
                    **identity,
                    "instance_id": inst["instance_id"],
                    "tokens": tokens,
                    "behavior_logprobs": behavior,
                    "mask": mask,
                    "turns": turns,
                    "termination": termination,
                },
                f,
            )
        patch = environments.patch(sb)
    while mask and not mask[-1]:
        tokens.pop()
        behavior.pop()
        mask.pop()
    episode = {
        **identity,
        "instance_id": inst["instance_id"],
        "tokens": tokens,
        "behavior_logprobs": behavior,
        "mask": mask,
        "turns": turns,
        "termination": termination,
        "generated_tokens": sum(mask),
        "modal_image_id": environments.image(inst).object_id,
    }
    (path / "prediction.patch").write_text(patch)
    with gzip.open(path / "ungraded.json.gz", "wt") as f:
        json.dump(episode, f)
    ready = time.time()
    log.event(
        "rollout_ready",
        **identity,
        instance_id=inst["instance_id"],
        generation_s=ready - start,
        generated_tokens=sum(mask),
    )
    return {
        "path": str(path),
        "identity": identity,
        "started_at": start,
        "ready_at": ready,
        "generation_s": ready - start,
    }


def grade_rollout(prepared, inst, environments, log):
    path, identity = Path(prepared["path"]), prepared["identity"]
    grade_start = time.time()
    queue_s = grade_start - prepared["ready_at"]
    log.event(
        "grading_start",
        **identity,
        instance_id=inst["instance_id"],
        grading_queue_s=queue_s,
    )
    verdict = environments.grade(inst, (path / "prediction.patch").read_text(), path)
    episode = load_episode(path / "ungraded.json.gz")
    episode.update(
        reward=verdict["reward"],
        elapsed_s=time.time() - prepared["started_at"],
        generation_s=prepared["generation_s"],
        grading_queue_s=queue_s,
        grading_s=time.time() - grade_start,
    )
    trace = path / "trajectory.json.gz"
    with gzip.open(trace, "wt") as f:
        json.dump(episode, f)
    log.event(
        "episode",
        **identity,
        instance_id=inst["instance_id"],
        reward=episode["reward"],
        generated_tokens=episode["generated_tokens"],
        context_tokens=len(episode["tokens"]),
        termination=episode["termination"],
        generation_s=episode["generation_s"],
        grading_queue_s=queue_s,
        elapsed_s=episode["elapsed_s"],
        grading_s=episode["grading_s"],
        trace=str(trace),
    )
    return {
        "path": str(trace),
        "reward": episode["reward"],
        "generated_tokens": episode["generated_tokens"],
        "termination": episode["termination"],
    }


class GradingPool:
    def __init__(self, workers, queue_size):
        self.slots = threading.BoundedSemaphore(workers + queue_size)
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix="swe-grading")

    def submit(self, *args):
        self.slots.acquire()
        try:
            future = self.pool.submit(*args)
        except BaseException:
            self.slots.release()
            raise
        future.add_done_callback(lambda _: self.slots.release())
        return future

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.pool.shutdown(wait=True)


def load_episode(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)
