# Parallel ML Research

Use this skill when running ML experiments that can be parallelized across sandbox subagents.

---

## Phase 1: Plan before spawning

Before spawning any subagent, spend time reasoning through the full experiment space:

- List all experiments you intend to run.
- Group them by dependency: what must be sequential vs. what is truly parallel?
- Identify which experiments are cheap/fast (ablations, hyperparameter sweeps) vs. expensive (full training runs).
- Order your launch plan: cheap, fast experiments first - they give early signal to prune the expensive ones.
- Estimate relative wall-clock time for each job. Track this mentally and check in on fast jobs sooner.

Use `set_reasoning_effort("high")` before this planning step - it pays off. Reasoning resets after each LLM call so you only pay for it once.

---

## Phase 2: Base image setup (always do this first)

Environment setup (installing deps, cloning repos, downloading data) is expensive repeated work. Eliminate it:

1. Spawn a single **setup agent** on a GPU instance (so GPU drivers and CUDA deps install correctly).
2. Give it a focused task: install all dependencies, clone repos, download datasets, run a minimal smoke test.
3. Once it reports success, call `snapshot_image` to freeze the filesystem.
4. Delete the setup agent.
5. Spawn all experiment agents using `image_id` from the snapshot - they start with everything ready.

> Always set up on a GPU instance even if some later experiments run on CPU, since GPU driver setup is the most fragile part.

---

## Phase 3: Launch experiments aggressively in parallel

- Spawn all independent experiments simultaneously before awaiting any result.
- Use `invoke_subagent` for each (non-blocking), then `wait_for_first_subagent_result` to process results as they arrive.
- Don't wait for slow jobs before launching fast ones - get everything in flight.
- Give each subagent a focused, scoped objective. They are fully isolated: no shared filesystem, no shared history.

**Subagent objective writing:**
- Role-defining, not task-defining: describe who the agent is and what it owns.
- Self-contained: include all context (repo path, dataset path, evaluation metric, what to report back).
- Scoped: one experiment per agent, not the whole problem.

**Subagent task (sent via invoke_subagent):**
- Specific, actionable, and includes success criteria.
- Include what format to report results in (e.g., "report final val BPB and training time").

---

## Phase 4: Lightweight experimentation principles

- **Start small**: run experiments at reduced scale (fewer steps, smaller model, subset of data) to validate the setup before committing to full runs.
- **Ablate one variable at a time**: don't change architecture and optimizer simultaneously.
- **Fail fast**: if a job errors in the first minute, kill it, fix it, and relaunch rather than waiting for a full run.
- **Snapshot good states**: after any significant setup or midpoint you'd want to reuse, snapshot the image with a descriptive label.

---

## Phase 5: Tracking and prioritization

- After launching, use `list_subagents` to see status at a glance.
- Process results from fast jobs first (`wait_for_first_subagent_result`) and use early signal to:
  - Cancel experiments that are clearly dominated.
  - Spawn follow-up experiments on promising directions.
- When a slow job is running and you have nothing to wait on, use `set_reasoning_effort("high")` to reason about what the current results mean and what to run next.

---

## When to use high reasoning

- At the start of a new task, before planning experiments.
- After receiving a batch of results, before deciding next steps.
- When debugging a surprising failure.
- **Not** for routine tool calls (spawning, waiting, status updates).

---

## Explore agent pattern

Keep one general-purpose **explore agent** alive throughout the session. Use it for:
- Inspecting repo structure, config files, logs.
- Searching the web with curl.
- Spot-checking intermediate outputs from other agents.

This keeps state centralized and avoids redundant setup across throwaway agents.

---

## Summary checklist

1. [ ] Reason through the full experiment space (`set_reasoning_effort("high")`)
2. [ ] Spawn setup agent on GPU, install deps, snapshot image, delete agent
3. [ ] Spawn explore agent from snapshot
4. [ ] Spawn all independent experiment agents from snapshot simultaneously
5. [ ] Invoke all agents before awaiting any
6. [ ] Process results as they arrive; prune losers, double down on winners
7. [ ] Synthesize and call `finish()` only after all needed results are collected
