"""Main eval runner - Modal app for A/B testing documentation variants.

Runs coding agents (Claude Code CLI or Codex CLI) inside Modal Sandboxes
with documentation mounted as files. The agents generate code based only
on the provided docs, which is then evaluated for correctness.

Usage:
    # Run eval with current llms.txt using Claude Code
    modal run internal/eval/runner.py

    # Run with specific docs variant and agent
    modal run internal/eval/runner.py --agent claude --docs-variant llms_txt

    # Run specific tasks only
    modal run internal/eval/runner.py --task-ids hello_world,get_started

    # Compare two doc variants
    modal run internal/eval/runner.py \
        --docs-variant llms_txt \
        --docs-variant llms_txt_v2

    # Skip LLM judge (faster, pattern-only scoring)
    modal run internal/eval/runner.py --no-judge

    # Use Codex CLI agent
    modal run internal/eval/runner.py --agent codex
"""

import json
import time
from pathlib import Path

import modal

from .agents import AgentConfig, run_agent_in_sandbox
from .evaluator import evaluate_code
from .tasks import (
    EvalResult,
    EvalTask,
    fetch_docs_from_url,
    load_all_tasks,
    load_docs_variant,
)

app = modal.App("docs-eval-runner")

# Image for the evaluator (judge) - needs LLM API access for scoring
eval_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "anthropic",
    "openai",
    "pyyaml",
)

LLMS_TXT_URL = "https://modal.com/llms.txt"


@app.function(
    image=eval_image,
    secrets=[
        modal.Secret.from_name("anthropic-secret", required_keys=["ANTHROPIC_API_KEY"]),
        modal.Secret.from_name("openai-secret", required_keys=["OPENAI_API_KEY"]),
    ],
    timeout=600,
)
def run_single_task(
    task_dict: dict,
    docs_content: str,
    agent_type: str,
    model: str | None,
    use_judge: bool,
    judge_agent: str,
    block_network: bool = True,
) -> dict:
    """Run a single eval task: spawn a coding agent in a sandbox, then evaluate."""
    task = EvalTask.from_dict(task_dict)
    config = AgentConfig(
        agent_type=agent_type, model=model, block_network=block_network
    )

    # Run the coding agent inside a Modal Sandbox
    try:
        generated_code, generation_time = run_agent_in_sandbox(
            task_description=task.description,
            docs_content=docs_content,
            config=config,
            app=app,
        )
    except Exception as e:
        return EvalResult(
            task_id=task.id,
            docs_variant="",
            agent=agent_type,
            model=config.resolved_model,
            generated_code="",
            scores={},
            overall_score=0.0,
            error=f"Agent failed: {e}",
        ).to_dict()

    # Evaluate the generated code
    eval_start = time.time()
    eval_scores = evaluate_code(
        generated_code=generated_code,
        reference_code=task.reference_code,
        use_judge=use_judge,
        judge_agent=judge_agent,
    )
    eval_time = time.time() - eval_start

    scores = eval_scores.to_dict()
    scores["generation_time_s"] = round(generation_time, 2)
    scores["eval_time_s"] = round(eval_time, 2)

    return EvalResult(
        task_id=task.id,
        docs_variant="",
        agent=agent_type,
        model=config.resolved_model,
        generated_code=generated_code,
        scores=scores,
        overall_score=eval_scores.overall_score,
    ).to_dict()


def print_report(
    results_by_variant: dict[str, list[dict]],
) -> None:
    """Print a comparison report across doc variants."""
    print("\n" + "=" * 80)
    print("DOCS EVAL REPORT")
    print("=" * 80)

    for variant, results in results_by_variant.items():
        print(f"\n--- Variant: {variant} ---")

        total_score = 0.0
        syntax_pass = 0
        n = len(results)

        for r in results:
            task_id = r["task_id"]
            score = r["overall_score"]
            total_score += score
            syntax_ok = r["scores"].get("syntax_valid", False)
            if syntax_ok:
                syntax_pass += 1

            status = "PASS" if syntax_ok else "FAIL"
            judge_info = ""
            if "judge" in r["scores"]:
                j = r["scores"]["judge"]
                judge_info = (
                    f" | API:{j['api_correctness']}/5"
                    f" Func:{j['functional_match']}/5"
                    f" Quality:{j['code_quality']}/5"
                )

            error_info = ""
            if r.get("error"):
                error_info = f" ERROR: {r['error']}"

            print(
                f"  {task_id:30s} [{status}] score={score:.2f}{judge_info}{error_info}"
            )

        avg_score = total_score / n if n > 0 else 0
        print(f"\n  Summary: {syntax_pass}/{n} syntax valid, avg score={avg_score:.3f}")

    # Side-by-side comparison if multiple variants
    if len(results_by_variant) > 1:
        print("\n--- COMPARISON ---")
        variants = list(results_by_variant.keys())

        # Collect task IDs from all variants
        all_task_ids = set()
        for results in results_by_variant.values():
            for r in results:
                all_task_ids.add(r["task_id"])

        header = f"{'Task':30s}"
        for v in variants:
            header += f" | {v:>15s}"
        print(header)
        print("-" * len(header))

        for task_id in sorted(all_task_ids):
            row = f"{task_id:30s}"
            for v in variants:
                score = 0.0
                for r in results_by_variant[v]:
                    if r["task_id"] == task_id:
                        score = r["overall_score"]
                        break
                row += f" | {score:>15.3f}"
            print(row)

        print("-" * len(header))
        row = f"{'AVERAGE':30s}"
        for v in variants:
            results = results_by_variant[v]
            avg = (
                sum(r["overall_score"] for r in results) / len(results)
                if results
                else 0
            )
            row += f" | {avg:>15.3f}"
        print(row)

    print("\n" + "=" * 80)


@app.local_entrypoint()
def main(
    agent: str = "claude",
    model: str | None = None,
    docs_variant: list[str] | None = None,
    docs_url: str | None = None,
    task_ids: str | None = None,
    category: str | None = None,
    no_judge: bool = False,
    judge_agent: str = "claude",
    no_network_isolation: bool = False,
    output: str | None = None,
):
    """Run the docs eval framework.

    Args:
        agent: Coding agent CLI to use ("claude" or "codex")
        model: Model override (uses agent default if not set)
        docs_variant: Docs variant name(s) from internal/eval/docs/
        docs_url: URL to fetch docs from (default: modal llms.txt)
        task_ids: Comma-separated task IDs to run (default: all)
        category: Filter tasks by category
        no_judge: Skip LLM-as-judge evaluation
        judge_agent: Agent to use for judging ("claude" or "codex")
        no_network_isolation: Disable network isolation (allow all outbound)
        output: Path to save JSON results
    """
    # Load tasks
    task_id_list = task_ids.split(",") if task_ids else None
    category_list = [category] if category else None
    tasks = load_all_tasks(task_ids=task_id_list, categories=category_list)

    if not tasks:
        print("No tasks found! Run `modal run internal/eval/generate_tasks.py` first.")
        return

    print(f"Loaded {len(tasks)} eval tasks")

    # Determine doc variants to test
    variants_to_test: dict[str, str] = {}

    if docs_variant:
        for v in docs_variant:
            variants_to_test[v] = load_docs_variant(v)
    elif docs_url:
        variants_to_test[docs_url] = fetch_docs_from_url(docs_url)
    else:
        # Default: fetch current llms.txt
        print(f"Fetching docs from {LLMS_TXT_URL}...")
        variants_to_test["llms_txt_current"] = fetch_docs_from_url(LLMS_TXT_URL)

    print(
        f"Testing {len(variants_to_test)} doc variant(s): {list(variants_to_test.keys())}"
    )
    block_network = not no_network_isolation
    print(f"Agent: {agent}" + (f" ({model})" if model else ""))
    print(f"Judge: {'disabled' if no_judge else judge_agent}")
    print(f"Network isolation: {'enabled' if block_network else 'disabled'}")

    # Run evaluations
    all_results: dict[str, list[dict]] = {}

    for variant_name, docs_content in variants_to_test.items():
        print(f"\nRunning eval with docs variant: {variant_name}")

        # Launch all tasks in parallel using starmap
        task_dicts = [t.to_dict() for t in tasks]

        results = list(
            run_single_task.starmap(
                [
                    (
                        td,
                        docs_content,
                        agent,
                        model,
                        not no_judge,
                        judge_agent,
                        block_network,
                    )
                    for td in task_dicts
                ]
            )
        )

        # Tag results with variant name
        for r in results:
            r["docs_variant"] = variant_name

        all_results[variant_name] = results

    # Print report
    print_report(all_results)

    # Save results
    if output:
        output_path = Path(output)
        flat_results = []
        for results in all_results.values():
            flat_results.extend(results)

        with open(output_path, "w") as f:
            json.dump(flat_results, f, indent=2)
        print(f"\nResults saved to {output_path}")
