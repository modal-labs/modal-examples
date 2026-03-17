# ---
# cmd: ["modal", "run", "06_gpu_and_ml/agents/autokernel.py", "--dry-run"]
# ---

# # AutoKernel: hire a kernel engineer for $10 on Modal

# GPU kernel optimization is a specialized skill in computing.
# A kernel engineer spends days hand-tuning memory access
# patterns, tile sizes, and warp configurations to squeeze performance out of a GPU.

# [AutoKernel](https://github.com/RightNow-AI/autokernel) automates this.
# Inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch),
# it gives an AI coding agent a PyTorch model and lets it run hundreds of
# optimization experiments overnight — editing Triton kernels, benchmarking,
# keeping improvements, reverting failures. Each experiment takes ~90 seconds.

# In this example, we deploy AutoKernel on Modal with an H100 GPU and
# Claude Code as the autonomous agent. One command, go to sleep, wake up to
# optimized kernels and a full experiment log like this:

#  | # | What the agent tried | TFLOPS | vs PyTorch | Result |
# |---|---|---|---|---|
# | 0 | Unmodified baseline (has NaN bug in softmax) | 223 | 8.5x | ❌ NaN in numerical stability check |
# | 1 | Fix NaN: compute QK dot product in fp16 to match reference overflow behavior | 267 | **10.2x** | ✅ Kept — first correct kernel |
# | 2 | Increase tile size: BLOCK_M 64→128 for more work per thread block | 199 | 7.6x | ✅ Reverted — register pressure hurt |
# | 3 | Software pipelining: num_stages=2 to overlap memory loads with compute | 251 | 9.6x | ✅ Reverted — slower than baseline |
# | 4 | More parallelism: num_warps 4→8 per thread block | 186 | 7.1x | ✅ Reverted — worse occupancy |
# | 5 | Autotune: let Triton search over BLOCK_M, BLOCK_N, and num_warps configs | 275 | **10.5x** | ✅ Kept — Triton picked best combo |
# | 6 | Extend autotune: add BLOCK_N=128 candidates to the search space | 299 | **11.4x** | ✅ Kept — larger KV blocks won |
# | 7 | FlashAttention-v2 split: separate masked/unmasked loops, skip causal mask on non-diagonal blocks | 339 | **13.0x** | ✅ Kept — major algorithmic win |
# | 8 | Faster math: replace exp() with exp2() using log2(e) scaling | 377 | 14.3x | ❌ Reverted — edge case correctness failure |
# **223 → 339 TFLOPS** (1.52x improvement, 29% → 45% of H100 peak).

# ## How AutoKernel works

# The system has two parts: **AutoKernel** and **the agent**.

# AutoKernel provides the profiler, benchmarking capabilities, correctness checks, and an orchestrator.
# The agent (Claude Code in this case) decides what optimizations
# to try, writing actual Triton kernel code, and interpreting benchmark results.

# The pipeline has 5 stages:

# ```text
#                profile.py           extract.py        agent + bench.py      verify.py
# PyTorch    ──>  Rank kernels  ──>  Generate       ──>  Optimize each  ──>  End-to-end
#   model         by GPU time       baseline Triton     kernel (loop)       verification
# ```

# The agent loop works for each kernel as follows:
# 1. **Edit** `kernel.py` — try an optimization (tiling, block sizes, memory layout, etc.)
# 2. **Bench** — run `bench.py` which does 5-stage correctness checks + roofline analysis
# 3. **Decide** — if PASS and faster, keep. If FAIL or slower, `git revert`. Repeat.

# The orchestrator uses Amdahl's law to decide which kernel to optimize next. For instance,
# a 1.5x speedup on a 60% kernel beats a 3x speedup on a 5% kernel.

# ## Setup

# We need two things in our container: AutoKernel and the Claude Code CLI.

import modal
import os
import time

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @anthropic-ai/claude-code",
    )
    .run_commands(
        "git clone https://github.com/RightNow-AI/autokernel.git /root/autokernel",
        "cd /root/autokernel && /root/.local/bin/uv sync",
    )
)

app = modal.App("autokernel", image=image)

# Results persist on a Modal Volume so they survive after the GPU shuts down.
# We can check them by running `modal run autokernel.py --download`.

vol = modal.Volume.from_name("autokernel-results", create_if_missing=True)

# ## Model presets

# AutoKernel ships with self-contained model definitions — no HuggingFace download needed.
# The compact LLaMA (124M params, 12 heads, 768 hidden dim) is the default.
# When the profiler runs, it finds flash attention (5.8% of GPU time),
# matmul (4.9%, 4.5%, 3.7%), and reduce (3.0%) as the top bottlenecks.
# AutoKernel can optimize ~35% of total GPU time across these kernel types.


MODELS = {
    "llama": {
        "file": "models/llama_7b.py",
        "class_name": "LlamaModel",
        "input_shape": "1,512",
        "dtype": "float16",
    },
    "llama-7b": {
        "file": "models/llama_7b.py",
        "class_name": "LlamaModel7B",
        "input_shape": "1,2048",
        "dtype": "float16",
    },
    "bert": {
        "file": "models/bert_base.py",
        "class_name": "BertModel",
        "input_shape": "8,512",
        "dtype": "float16",
    },
}

UV = "/root/.local/bin/uv"
WORKDIR = "/root/autokernel"

# ## Helpers

# A thin wrapper around `subprocess.run` that prints output and raises on failure.


def run_step(name: str, cmd: str):
    import subprocess

    print(f"\n{'='*60}\n  {name}\n{'='*60}\n$ {cmd}\n")
    start = time.time()
    result = subprocess.run(
        cmd, shell=True, cwd=WORKDIR, text=True, capture_output=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    elapsed = time.time() - start
    print(f"\n[{name}] done in {elapsed:.1f}s (exit {result.returncode})")
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {name}")
    return result.stdout


def save_results(model_name: str):
    """Copy artifacts to the persistent volume."""
    import shutil

    dst = f"/results/{model_name}/{int(time.time())}"
    os.makedirs(dst, exist_ok=True)
    for fname in ["results.tsv", "progress.png", "kernel.py"]:
        src = os.path.join(WORKDIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  saved {fname}")
    ws_src = os.path.join(WORKDIR, "workspace")
    if os.path.exists(ws_src):
        shutil.copytree(ws_src, os.path.join(dst, "workspace"))
        print(f"  saved workspace/")
    vol.commit()
    print(f"\nResults saved to volume at /results/{model_name}/")


def count_experiments() -> int:
    tsv = os.path.join(WORKDIR, "results.tsv")
    if not os.path.exists(tsv):
        return 0
    with open(tsv) as f:
        return max(0, sum(1 for _ in f) - 1)


# ## The agent loop

# We launch Claude Code in headless mode
# using `-p` with `--allowedTools` to auto-approve
# file edits and bash commands.

# The agent reads AutoKernel's `program.md` — an instruction document
# that contains a 6-step GPU optimization playbook, decision framework, and
# Amdahl's law reasoning. It then enters the experiment loop autonomously.

# A background thread watches `results.tsv` for new rows and prints
# each experiment result as it lands — so you see progress in real-time.


def run_agent_loop(max_turns: int = 500):
    import subprocess
    import threading

    agent_prompt = (
        "Read program.md and start optimizing. "
        "The environment is already set up — prepare.py, profile.py, and extract.py have all been run. "
        "Workspace has extracted kernels. Run `uv run orchestrate.py next` to get the first kernel. "
        "Then enter the experiment loop: edit kernel.py, run bench.py, log to results.tsv, keep or revert. "
        "Maximize experiments per hour. Each cycle should be: edit, bench, log, decide — 4 tool calls max."
    )

    # Find the claude binary
    which = subprocess.run(
        "which claude", shell=True, capture_output=True, text=True
    )
    claude_bin = which.stdout.strip()
    if not claude_bin:
        for path in [
            "/root/.local/bin/claude",
            "/root/.claude/local/claude",
            "/usr/local/bin/claude",
        ]:
            if os.path.exists(path):
                claude_bin = path
                break

    if not claude_bin:
        print("ERROR: Claude CLI not found. Skipping agent loop.")
        return

    print(f"Claude binary: {claude_bin}")

    claude_cmd = [
        claude_bin,
        "-p", agent_prompt,
        "--allowedTools", "Bash,Read,Edit,Write,Glob,Grep",
        "--max-turns", str(max_turns),
        "--output-format", "stream-json",
        "--verbose",
    ]

    # Background watcher: prints each new results.tsv row as it appears
    results_tsv = os.path.join(WORKDIR, "results.tsv")
    start_time = time.time()
    seen_lines = 0
    stop_watcher = threading.Event()

    def watch_results():
        nonlocal seen_lines
        while not stop_watcher.is_set():
            if os.path.exists(results_tsv):
                with open(results_tsv) as f:
                    lines = f.readlines()
                for i, line in enumerate(lines):
                    if i == 0 and seen_lines == 0:
                        cols = line.strip().split("\t")
                        print(f"\n  {'─'*60}")
                        print(f"  columns: {', '.join(cols)}")
                        print(f"  {'─'*60}")
                        seen_lines = 1
                        continue
                    if i >= seen_lines:
                        cols = line.strip().split("\t")
                        elapsed = round((time.time() - start_time) / 60, 1)
                        if len(cols) >= 8:
                            exp_num = cols[0]
                            kernel = cols[2] if len(cols) > 2 else "?"
                            tflops = cols[3] if len(cols) > 3 else "?"
                            latency = cols[4] if len(cols) > 4 else "?"
                            pct_peak = cols[5] if len(cols) > 5 else "?"
                            speedup = cols[6] if len(cols) > 6 else "?"
                            correct = cols[7] if len(cols) > 7 else "?"
                            desc = cols[9] if len(cols) > 9 else ""
                            icon = "✅" if correct == "PASS" else "❌"
                            print(f"\n  {icon} EXPERIMENT #{exp_num} ({elapsed}min)")
                            print(f"     Kernel:  {kernel}")
                            print(f"     Speedup: {speedup} vs PyTorch")
                            print(f"     TFLOPS:  {tflops} ({pct_peak} peak)")
                            print(f"     Latency: {latency} us")
                            print(f"     Correct: {correct}")
                            if desc:
                                print(f"     Desc:    {desc[:80]}")
                        seen_lines = i + 1
            stop_watcher.wait(timeout=5)

    watcher = threading.Thread(target=watch_results, daemon=True)
    watcher.start()

    # Run the agent with restart logic (handles context window limits)
    max_restarts = 5
    for attempt in range(max_restarts):
        exp_count = seen_lines - 1 if seen_lines > 0 else 0
        print(
            f"\n{'='*60}\n"
            f"  Agent loop (attempt {attempt + 1}/{max_restarts}, {exp_count} experiments so far)\n"
            f"{'='*60}\n"
        )

        proc = subprocess.Popen(
            claude_cmd, cwd=WORKDIR,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        # Stream stdout, print compact agent activity
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                import json
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            if event.get("type") == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "tool_use":
                        name = block.get("name", "")
                        inp = block.get("input", {})
                        if name == "Bash":
                            print(f"  [Bash] {inp.get('command', '')[:120]}")
                        elif name in ("Edit", "Write"):
                            print(f"  [{name}] {inp.get('file_path', '?')}")
            elif event.get("type") == "result":
                cost = event.get("total_cost_usd", 0)
                turns = event.get("num_turns", 0)
                subtype = event.get("subtype", "")
                print(f"\n  Session: {turns} turns, ${cost:.2f}, {subtype}")

        proc.wait()
        stderr = proc.stderr.read()
        if stderr.strip():
            print(f"  STDERR: {stderr[:500]}")

        elapsed = time.time() - start_time
        exp_count = seen_lines - 1 if seen_lines > 0 else 0
        print(f"\nAgent ran for {elapsed/60:.1f} total minutes")

        if proc.returncode == 0:
            print(f"Agent finished. {exp_count} experiments logged.")
            break
        if attempt > 0 and exp_count == 0:
            print("No experiments logged — stopping.")
            break
        print("Agent exited early — restarting...")

    stop_watcher.set()
    watcher.join(timeout=2)
    exp_count = seen_lines - 1 if seen_lines > 0 else 0
    print(f"\nAgent loop complete. {exp_count} experiments in results.tsv.")


# ## The full pipeline

# One Modal function that runs the entire overnight optimization.
# Steps 1-3 are AutoKernel infrastructure (profile, extract, prepare).
# Step 4 is the agent loop where the experiments happen.
# Step 5 plugs optimized kernels back into the model for end-to-end verification.

# The `anthropic-secret` Modal secret provides the `ANTHROPIC_API_KEY`
# for Claude Code. Create it with:
# ```bash
# modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...
# ```


# We use `required_keys=[]` so the secret doesn't block dry-run mode.
# Claude Code picks up ANTHROPIC_API_KEY from the environment automatically.

@app.function(
    gpu="H100",
    timeout=36000,  # 10 hours
    volumes={"/results": vol},
    secrets=[modal.Secret.from_name("anthropic-secret", required_keys=[])],
)
def run_overnight(
    model_name: str = "llama",
    top_k: int = 5,
    backend: str = "triton",
    dry_run: bool = False,
    max_agent_turns: int = 500,
):
    cfg = MODELS[model_name]

    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║         AutoKernel on Modal — Overnight Run         ║
    ╠══════════════════════════════════════════════════════╣
    ║  Model:    {model_name:<40} ║
    ║  Backend:  {backend:<40} ║
    ║  Top-K:    {top_k:<40} ║
    ║  Dry run:  {str(dry_run):<40} ║
    ║  Agent:    {'SKIPPED' if dry_run else f'Claude Code ({max_agent_turns} turns)':<40} ║
    ╚══════════════════════════════════════════════════════╝
    """)

    # Step 1: Generate test data and PyTorch baselines
    run_step("Step 1/5 · Prepare test data", f"{UV} run prepare.py")

    # Step 2: Profile the model — find which GPU kernels are bottlenecks
    profile_cmd = (
        f"{UV} run profile.py"
        f" --model {cfg['file']}"
        f" --class-name {cfg['class_name']}"
        f" --input-shape {cfg['input_shape']}"
        f" --dtype {cfg['dtype']}"
    )
    run_step("Step 2/5 · Profile model", profile_cmd)

    # Step 3: Extract top-K bottleneck kernels as standalone Triton files
    run_step(
        "Step 3/5 · Extract kernels",
        f"{UV} run extract.py --top {top_k} --backend {backend}",
    )

    # Step 4: The agent optimization loop (or a single bench in dry-run mode)
    if dry_run:
        run_step("Step 4/5 · Bench (single run)", f"{UV} run bench.py")
    else:
        run_agent_loop(max_turns=max_agent_turns)

    # Step 5: Verify end-to-end correctness (only if agent produced optimized kernels)
    optimized_dir = os.path.join(WORKDIR, "workspace")
    has_optimized = os.path.exists(optimized_dir) and any(
        f.endswith("_optimized.py")
        for f in os.listdir(optimized_dir)
        if os.path.isfile(os.path.join(optimized_dir, f))
    )

    if has_optimized:
        verify_cmd = (
            f"{UV} run verify.py"
            f" --model {cfg['file']}"
            f" --class-name {cfg['class_name']}"
            f" --input-shape {cfg['input_shape']}"
            f" --dtype {cfg['dtype']}"
        )
        run_step("Step 5/5 · Verify end-to-end", verify_cmd)
    else:
        print("\nStep 5/5 · Skipped — no optimized kernels yet\n")

    # Save everything to the persistent volume
    save_results(model_name)

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║                     DONE                            ║
    ╠══════════════════════════════════════════════════════╣
    ║  Results: modal volume get autokernel-results       ║
    ║  Or run:  modal run autokernel.py --download        ║
    ╚══════════════════════════════════════════════════════╝
    """)


# ## Check results

# A lightweight function that reads results from the volume to download them.


@app.function(volumes={"/results": vol})
def download_results(model_name: str = "llama"):
    base = f"/results/{model_name}"
    if not os.path.exists(base):
        print(f"No results found for {model_name}")
        return

    runs = sorted(os.listdir(base))
    if not runs:
        print("No runs found.")
        return

    latest = os.path.join(base, runs[-1])
    print(f"Latest run: {latest}\n")
    tsv = os.path.join(latest, "results.tsv")
    if os.path.exists(tsv):
        with open(tsv) as f:
            print(f.read())
    print("\nFiles:")
    for root, dirs, files in os.walk(latest):
        for fname in files:
            path = os.path.join(root, fname)
            print(f"  {os.path.relpath(path, latest)} ({os.path.getsize(path):,} bytes)")


# ## Run it

# Start with a dry run to verify the pipeline works end-to-end.
# This profiles the model, extracts kernels, and runs a single benchmark —
# no LLM API calls, no Anthropic secret needed.

# ```bash
# modal run autokernel.py --dry-run
# ```

# Once that passes, you can run this in various ways:

# For a small run with a few optimizations:

# ```bash
# modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...
# modal run autokernel.py --max-turns 50
# ```

# For the full overnight run:

# ```bash
# modal run autokernel.py
# ```

# Check results once it is done

# ```bash
# modal run autokernel.py --download
# ```

# # Main Modal entrypoint

@app.local_entrypoint()
def main(
    model: str = "llama",
    top_k: int = 5,
    backend: str = "triton",
    dry_run: bool = False,
    max_turns: int = 500,
    download: bool = False,
):
    if download:
        download_results.remote(model_name=model)
        return

    print(
        f"Launching AutoKernel on Modal "
        f"(model={model}, gpu=H100, backend={backend}, dry_run={dry_run})"
    )
    if not dry_run:
        print(f"Running for up to {max_turns} agent turns on an H100.")

    run_overnight.remote(
        model_name=model,
        top_k=top_k,
        backend=backend,
        dry_run=dry_run,
        max_agent_turns=max_turns,
    )