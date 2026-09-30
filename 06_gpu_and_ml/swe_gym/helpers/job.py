"""Run rollout, grading, and training workers for one LoRA client."""

import math
import queue
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from itertools import batched

import numpy as np
from tinker import types

from .environments import InfrastructureError
from .rollouts import GradingPool, generate_rollout, grade_rollout
from .training import atomic_json, prepare_minibatch


def publish_policy(trainer, service, name):
    receipt = trainer.save_weights_for_sampler(name=name).result(timeout=3600)
    return service.create_sampling_client(model_path=receipt.path)


class TrainingJob:
    def __init__(
        self, job, trainer, service, tokenizer, environments, tasks, cfg, log, reference
    ):
        self.job, self.trainer, self.service = job, trainer, service
        self.tokenizer, self.env, self.tasks = tokenizer, environments, tasks
        self.cfg, self.log, self.reference = cfg, log, reference
        self.lock = threading.Lock()
        self.version = 0
        self.sampler = None

    def run(self, phase, deadline, start_step=0, start_group=0):
        if start_step >= self.cfg["steps"]:
            return
        self.version = start_step
        self.start_group = start_group
        if self.sampler is None:
            self.sampler = publish_policy(
                self.trainer, self.service, f"{phase}-job{self.job}-initial"
            )
        output = queue.Queue(self.cfg["completed_group_buffer"])
        stop = threading.Event()
        producer = threading.Thread(
            target=self.producer, args=(phase, output, stop), daemon=True
        )
        producer.start()
        try:
            for step in range(start_step, self.cfg["steps"]):
                # Collect mixed-reward groups no more than one publication old (one step off policy)
                chosen, incoming, errors = [], [], 0
                while len(chosen) < self.cfg["groups_per_batch"]:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise TimeoutError("Training wall time limit reached")
                    try:
                        future = output.get(timeout=remaining)
                    except queue.Empty:
                        raise TimeoutError("Training wall time limit reached") from None
                    try:
                        group = future.result()
                    except Exception as exc:
                        self.log.event("group_failed", job=self.job, error=str(exc))
                        errors += 1
                        if errors >= 5:
                            raise
                        continue
                    incoming.extend(episode["reward"] for episode in group["episodes"])
                    if group["mixed"] and self.version - group["policy_version"] <= 1:
                        chosen.append(group)
                for minibatch in batched(chosen, self.cfg["minibatch_groups"]):
                    data = prepare_minibatch(
                        minibatch, self.reference, self.cfg["kl_coef"]
                    )
                    fwd = self.trainer.forward_backward(
                        data,
                        "ppo",
                        loss_fn_config={
                            "clip_low_threshold": 0.8,
                            "clip_high_threshold": 1.28,
                        },
                    )
                    opt = self.trainer.optim_step(
                        types.AdamParams(learning_rate=1e-6, grad_clip_norm=1.0)
                    )
                    result = fwd.result(timeout=7200)
                    opt.result(timeout=7200)
                    if not all(
                        math.isfinite(float(x)) for x in result.metrics.values()
                    ):
                        raise InfrastructureError("Nonfinite training metrics")
                name = f"{phase}-job{self.job}-step{step + 1:04d}"
                state = self.trainer.save_state(name).result(timeout=3600)
                sampler = publish_policy(self.trainer, self.service, name)
                with self.lock:
                    self.sampler = sampler
                    self.version += 1
                row = self.log.event(
                    "step_done",
                    job=self.job,
                    step=step,
                    reward_mean=float(np.mean(incoming)),
                    checkpoint=state.path,
                )
                atomic_json(self.log.root / f"train-job{self.job}-checkpoint.json", row)
        finally:
            stop.set()
            producer.join()

    def producer(self, phase, output, stop):
        order = list(self.tasks)
        random.Random(self.cfg["seed"]).shuffle(order)
        group = self.start_group
        limit = self.cfg["inflight_groups"]
        with (
            ThreadPoolExecutor(
                self.cfg["rollout_workers"], thread_name_prefix="swe-rollout"
            ) as episodes,
            GradingPool(
                self.cfg["grading_workers"], self.cfg["grading_queue_size"]
            ) as graders,
            ThreadPoolExecutor(limit) as groups,
        ):
            pending = set()
            while not stop.is_set() or pending:
                while not stop.is_set() and len(pending) < limit and not output.full():
                    with self.lock:
                        sampler, version = self.sampler, self.version
                    inst = order[group % len(order)]
                    pending.add(
                        groups.submit(
                            self.run_group,
                            inst,
                            phase,
                            group,
                            sampler,
                            version,
                            episodes,
                            graders,
                        )
                    )
                    group += 1
                if not pending:
                    stop.wait(0.2)
                    continue
                done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    while not stop.is_set():
                        try:
                            output.put(future, timeout=0.5)
                            break
                        except queue.Full:
                            pass

    def run_group(self, inst, phase, group, sampler, version, pool, graders):
        identity = dict(phase=phase, job=self.job, group=group, policy_version=version)
        futures = {
            pool.submit(
                generate_rollout,
                inst,
                sampler,
                self.tokenizer,
                self.env,
                self.cfg,
                self.log,
                {**identity, "attempt": attempt},
            )
            for attempt in range(self.cfg["group_size"])
        }
        graded = []
        try:
            for future in as_completed(futures):
                graded.append(
                    graders.submit(
                        grade_rollout, future.result(), inst, self.env, self.log
                    )
                )
            results = [future.result() for future in graded]
        finally:
            wait([*futures, *graded])
        rewards = [e["reward"] for e in results]
        mixed = 0 < sum(rewards) < len(rewards)
        return {
            "episodes": results,
            "policy_version": version,
            "group": group,
            "mixed": mixed,
        }
