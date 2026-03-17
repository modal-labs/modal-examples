"""
AutoKernel on Modal — One-click overnight GPU kernel optimization.

Spins up an H100, profiles a LLaMA model, extracts bottleneck kernels,
and lets Claude Agent SDK optimize them autonomously for hours.

Usage:
    # Dry run — profile + extract + bench (no agent, no API key needed)
    modal run autokernel_modal.py --dry-run

    # Full overnight run (needs ANTHROPIC_API_KEY in Modal secrets)
    modal run autokernel_modal.py

    # Short test run (5 agent turns)
    modal run autokernel_modal.py --max-turns 5

    # Check results the next morning
    modal run autokernel_modal.py --download
"""

import modal
import os
import time

# ---------------------------------------------------------------------------
# Image: AutoKernel + Claude Agent SDK
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    # uv — fast Python package manager
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    # Claude Code CLI (via npm)
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @anthropic-ai/claude-code",
    )
    # AutoKernel repo + deps
    .run_commands(
        "git clone https://github.com/RightNow-AI/autokernel.git /root/autokernel",
        "cd /root/autokernel && /root/.local/bin/uv sync",
    )
)

app = modal.App("autokernel", image=image)

# Persistent storage — survives after the GPU container dies
vol = modal.Volume.from_name("autokernel-results", create_if_missing=True)

# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
UV = "/root/.local/bin/uv"
WORKDIR = "/root/autokernel"


def run_step(name: str, cmd: str):
    """Run a shell command, stream output, raise on failure."""
    import subprocess

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}\n")
    print(f"$ {cmd}\n")

    start = time.time()
    result = subprocess.run(
        cmd, shell=True, cwd=WORKDIR, text=True, capture_output=True,
    )
    elapsed = time.time() - start

    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    print(f"\n[{name}] finished in {elapsed:.1f}s (exit code {result.returncode})")

    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {name}")

    return result.stdout


def save_results(model_name: str):
    """Copy all artifacts to the persistent volume."""
    import shutil

    dst = f"/results/{model_name}/{int(time.time())}"
    os.makedirs(dst, exist_ok=True)

    for fname in ["results.tsv", "progress.png", "kernel.py", "experiments.jsonl"]:
        src = os.path.join(WORKDIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  saved {fname}")

    ws_src = os.path.join(WORKDIR, "workspace")
    if os.path.exists(ws_src):
        shutil.copytree(ws_src, os.path.join(dst, "workspace"))
        print(f"  saved workspace/")

    vol.commit()
    print(f"\nAll results saved to volume: /results/{model_name}/")
    return dst


def count_experiments() -> int:
    """Count rows in results.tsv to track progress."""
    tsv = os.path.join(WORKDIR, "results.tsv")
    if not os.path.exists(tsv):
        return 0
    with open(tsv) as f:
        return max(0, sum(1 for _ in f) - 1)


# ---------------------------------------------------------------------------
# Agent loop using Claude Agent SDK (Python)
# ---------------------------------------------------------------------------
def run_agent_loop(max_turns: int = 500):
    """
    Launch Claude Code CLI in headless mode. Watches results.tsv
    in real-time for new experiment results from bench.py.
    """
    import subprocess
    import threading

    agent_prompt = (
        "Read program.md and start optimizing. "
        "The environment is already set up — prepare.py, profile.py, and extract.py have all been run. "
        "Workspace has extracted kernels. Run `uv run orchestrate.py next` to get the first kernel. "
        "Then enter the experiment loop: edit kernel.py, run bench.py, log to results.tsv, keep or revert. "
        "Maximize experiments per hour. Each cycle should be: edit, bench, log, decide — 4 tool calls max."
    )

    which = subprocess.run("which claude", shell=True, capture_output=True, text=True)
    claude_bin = which.stdout.strip()
    if not claude_bin:
        for path in ["/root/.claude/local/claude", "/usr/local/bin/claude", "/usr/bin/claude"]:
            if os.path.exists(path):
                claude_bin = path
                break
    print(f"Claude binary: {claude_bin or 'NOT FOUND'}")

    if not claude_bin:
        print("ERROR: Claude CLI not found. Skipping agent loop.")
        return

    claude_cmd = [
        claude_bin,
        "-p", agent_prompt,
        "--allowedTools", "Bash,Read,Edit,Write,Glob,Grep",
        "--max-turns", str(max_turns),
        "--output-format", "stream-json",
        "--verbose",
    ]

    results_tsv = os.path.join(WORKDIR, "results.tsv")
    start_time = time.time()

    # Background thread: watch results.tsv for new rows
    seen_lines = 0
    stop_watcher = threading.Event()

    def watch_results():
        nonlocal seen_lines
        while not stop_watcher.is_set():
            if os.path.exists(results_tsv):
                with open(results_tsv) as f:
                    lines = f.readlines()
                # Print any new lines (skip header)
                for i, line in enumerate(lines):
                    if i == 0 and seen_lines == 0:
                        # Print header once
                        cols = line.strip().split("\t")
                        print(f"\n  {'─'*80}")
                        print(f"  results.tsv columns: {', '.join(cols)}")
                        print(f"  {'─'*80}")
                        seen_lines = 1
                        continue
                    if i >= seen_lines:
                        cols = line.strip().split("\t")
                        elapsed = round((time.time() - start_time) / 60, 1)

                        # Parse TSV row: experiment, tag, kernel_type, throughput_tflops,
                        # latency_us, pct_peak, speedup_vs_pytorch, correctness, peak_vram_mb, description
                        if len(cols) >= 8:
                            exp_num = cols[0]
                            kernel = cols[2] if len(cols) > 2 else "?"
                            tflops = cols[3] if len(cols) > 3 else "?"
                            latency = cols[4] if len(cols) > 4 else "?"
                            pct_peak = cols[5] if len(cols) > 5 else "?"
                            speedup = cols[6] if len(cols) > 6 else "?"
                            correct = cols[7] if len(cols) > 7 else "?"
                            desc = cols[9] if len(cols) > 9 else ""

                            status_icon = "✅" if correct == "PASS" else "❌"

                            print(f"\n  {status_icon} EXPERIMENT #{exp_num} ({elapsed}min)")
                            print(f"     Kernel:    {kernel}")
                            print(f"     Speedup:   {speedup} vs PyTorch")
                            print(f"     TFLOPS:    {tflops} ({pct_peak} peak)")
                            print(f"     Latency:   {latency} us")
                            print(f"     Correct:   {correct}")
                            if desc:
                                print(f"     Desc:      {desc[:80]}")
                        else:
                            print(f"  [results.tsv] {line.strip()}")

                        seen_lines = i + 1

            stop_watcher.wait(timeout=5)  # check every 5 seconds

    watcher = threading.Thread(target=watch_results, daemon=True)
    watcher.start()

    # Run the agent
    max_restarts = 5
    for attempt in range(max_restarts):
        print(f"\n{'='*70}")
        print(f"  Step 4/5 · Agent loop (attempt {attempt + 1}/{max_restarts})")
        print(f"  Experiments so far: {seen_lines - 1 if seen_lines > 0 else 0}")
        print(f"{'='*70}\n")

        proc = subprocess.Popen(
            claude_cmd,
            cwd=WORKDIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
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

            # Print agent tool calls (compact)
            if event.get("type") == "assistant":
                content = event.get("message", {}).get("content", [])
                for block in content:
                    if block.get("type") == "tool_use":
                        name = block.get("name", "")
                        inp = block.get("input", {})
                        if name == "Bash":
                            print(f"  [Bash] {inp.get('command', '')[:120]}")
                        elif name in ("Edit", "Write"):
                            print(f"  [{name}] {inp.get('file_path', '?')}")

            # Print session result
            elif event.get("type") == "result":
                cost = event.get("total_cost_usd", 0)
                turns = event.get("num_turns", 0)
                subtype = event.get("subtype", "")
                print(f"\n  Session: {turns} turns, ${cost:.2f}, {subtype}")

        proc.wait()
        elapsed = time.time() - start_time

        if proc.stderr:
            stderr = proc.stderr.read()
            if stderr.strip():
                print(f"  STDERR: {stderr[:500]}")

        print(f"\nAgent ran for {elapsed/60:.1f} total minutes")

        exp_count = seen_lines - 1 if seen_lines > 0 else 0
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


# ---------------------------------------------------------------------------
# Main: full overnight pipeline
# ---------------------------------------------------------------------------
@app.function(
    gpu="H100",
    timeout=36000,  # 10 hours
    volumes={"/results": vol},
    secrets=[modal.Secret.from_name("anthropic-secret")],
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
    ║  Agent:    {'SKIPPED' if dry_run else f'Claude SDK ({max_agent_turns} turns)':<40} ║
    ╚══════════════════════════════════════════════════════╝
    """)

    # ── Step 1: Prepare test data + baselines ──────────────────────────
    run_step("Step 1/5 · Prepare test data", f"{UV} run prepare.py")

    # ── Step 2: Profile the model ──────────────────────────────────────
    profile_cmd = (
        f"{UV} run profile.py"
        f" --model {cfg['file']}"
        f" --class-name {cfg['class_name']}"
        f" --input-shape {cfg['input_shape']}"
        f" --dtype {cfg['dtype']}"
    )
    run_step("Step 2/5 · Profile model", profile_cmd)

    # ── Step 3: Extract top-K bottleneck kernels ───────────────────────
    run_step(
        "Step 3/5 · Extract kernels",
        f"{UV} run extract.py --top {top_k} --backend {backend}",
    )

    # ── Step 4: Agent optimization loop ────────────────────────────────
    if dry_run:
        print("\n" + "="*70)
        print("  DRY RUN — skipping agent, running bench once")
        print("="*70 + "\n")
        run_step("Bench (single run)", f"{UV} run bench.py")
    else:
        run_agent_loop(max_turns=max_agent_turns)

    # ── Step 5: End-to-end verification ────────────────────────────────
    optimized_dir = os.path.join(WORKDIR, "workspace")
    has_optimized = any(
        f.endswith("_optimized.py")
        for f in os.listdir(optimized_dir)
        if os.path.isfile(os.path.join(optimized_dir, f))
    ) if os.path.exists(optimized_dir) else False

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
        print("\n" + "="*70)
        print("  Step 5/5 · Skipped — no optimized kernels yet")
        print("="*70 + "\n")

    # ── Save everything ────────────────────────────────────────────────
    save_results(model_name)

    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║                     DONE                            ║
    ╠══════════════════════════════════════════════════════╣
    ║  Results: modal volume get autokernel-results       ║
    ║  Or run:  modal run autokernel_modal.py --download  ║
    ╚══════════════════════════════════════════════════════╝
    """)


# ---------------------------------------------------------------------------
# Helper: download results
# ---------------------------------------------------------------------------
@app.function(volumes={"/results": vol})
def download_results(model_name: str = "llama"):
    """Print results.tsv from the most recent run."""
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
    else:
        print("No results.tsv found.")

    print(f"\nFiles:")
    for root, dirs, files in os.walk(latest):
        for fname in files:
            path = os.path.join(root, fname)
            size = os.path.getsize(path)
            print(f"  {os.path.relpath(path, latest)} ({size:,} bytes)")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
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

    print(f"Launching AutoKernel on Modal...")
    print(f"  Model:    {model}")
    print(f"  GPU:      H100")
    print(f"  Backend:  {backend}")
    print(f"  Dry run:  {dry_run}")
    print()

    if not dry_run:
        print("This will run for several hours on an H100.")
        print("Estimated cost: ~$30-45 (GPU + API calls)")
        print()

    run_overnight.remote(
        model_name=model,
        top_k=top_k,
        backend=backend,
        dry_run=dry_run,
        max_agent_turns=max_turns,
    )