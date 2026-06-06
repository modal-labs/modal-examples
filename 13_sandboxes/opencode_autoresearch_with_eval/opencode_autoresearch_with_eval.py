# ---
# cmd: ["python", "13_sandboxes/opencode_autoresearch_with_eval/opencode_autoresearch_with_eval.py", "--smoke-test", "--timeout", "10m", "--agent-timeout", "2m"]
# pytest: false
# ---

# # Run an OpenCode agent on an optimization task

# This example shows a small pattern for running coding agents on Modal. One
# [Sandbox](https://modal.com/docs/guide/sandbox) runs the agent, while a second
# Sandbox runs an HTTP verifier.
#
# The agent gets a small optimization task, edits a starting solution, and submits
# candidate answers to the verifier. When the agent exits, the local script asks
# the verifier for every submitted solution and prints the best one.
#
# ```text
#  local script
#      |
#      | starts both Sandboxes
#      v
#  +------------------+        HTTP submit        +--------------------+
#  | OpenCode Sandbox | ------------------------> | Verifier Sandbox   |
#  | edits solution.py|                           | grades candidates  |
#  +------------------+        submissions        +--------------------+
#      ^                                                   |
#      |                                                   |
#      +---------------- fetch final ranked results <------+
# ```
#
# The task is inspired by the `autocorrelation_first` task from
# [SimpleTES](https://github.com/wq-will/SimpleTES/tree/main/datasets/autocorrelation/autocorrelation_first):
# find a non-negative step function with a small discrete autoconvolution score.

# ## Package the task files

# The files that the agent and verifier need live beside this example in a nested
# asset directory. Keeping them as files makes the workflow easier to inspect than
# embedding a long prompt or evaluator in this script.

import argparse
import json
import shlex
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import modal
from modal.container_process import ContainerProcess

MINUTES = 60
GRADER_PORT = 8000
REMOTE_TASK_DIR = "/root/task"
LOCAL_TASK_DIR = Path(__file__).parent / "autocorrelation_first"

DEFAULT_MODEL = "openai/gpt-5.5"
DEFAULT_OPENAI_SECRET = "openai-secret"
DEFAULT_TRAJECTORY_FILE = Path("opencode_agent_trajectory.txt")
DEFAULT_BEST_SOLUTION_FILE = Path("opencode_best_solution.json")


def define_grader_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install("fastapi[standard]~=0.115.14")
        .add_local_dir(LOCAL_TASK_DIR, remote_path=REMOTE_TASK_DIR)
    )


# The agent Image installs OpenCode and gets the same task files. OpenCode reads
# `OPENAI_API_KEY` from the Modal Secret passed to the process.


def define_agent_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("curl", "git")
        .run_commands("curl -fsSL https://opencode.ai/install | bash")
        .env(
            {
                "PATH": "/root/.opencode/bin:${PATH}",
                "OPENCODE_DISABLE_AUTOUPDATE": "1",
            }
        )
        .add_local_dir(LOCAL_TASK_DIR, remote_path=REMOTE_TASK_DIR)
    )


# ## Start the verifier sidecar

# The verifier exposes three small HTTP endpoints:
# - `GET /task` returns the task text
# - `POST /submit` grades and records a candidate solution
# - `GET /submissions` returns the ranked submission list


def start_grader_sandbox(
    app: modal.App, image: modal.Image, timeout: int
) -> tuple[modal.Sandbox, str]:
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            "python",
            "-m",
            "uvicorn",
            "evaluate:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(GRADER_PORT),
            app=app,
            image=image,
            encrypted_ports=[GRADER_PORT],
            timeout=timeout,
            workdir=REMOTE_TASK_DIR,
        )

    url = sandbox.tunnels()[GRADER_PORT].url
    wait_for_http(f"{url}/health")
    return sandbox, url


def wait_for_http(url: str, timeout: int = 60) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            time.sleep(1)

    raise TimeoutError(f"Timed out waiting for {url}")


# ## Ask OpenCode to improve the solution

# `opencode run` is the non-interactive OpenCode mode. We use
# `--dangerously-skip-permissions` because this runs in a short-lived Sandbox
# that contains only the task files.


def build_agent_instruction(grader_url: str, max_submissions: int) -> str:
    return f"""
You are in {REMOTE_TASK_DIR}. Your goal is to improve solution.py for the
autocorrelation task.

Read autocorrelation_first.txt, then edit only the code between
# EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END in solution.py.

Use this command to grade a candidate:

    python submit.py

The GRADER_URL environment variable is already set to:

    {grader_url}

Submit at least one valid solution. Try at most {max_submissions} submissions, keep
the code simple, and stop once you have a good score.
""".strip()


def start_agent_sandbox(
    app: modal.App, image: modal.Image, timeout: int
) -> modal.Sandbox:
    with modal.enable_output():
        return modal.Sandbox.create(
            app=app,
            image=image,
            timeout=timeout,
            workdir=REMOTE_TASK_DIR,
        )


def run_agent(
    sandbox: modal.Sandbox,
    grader_url: str,
    model: str,
    openai_secret_name: str,
    max_submissions: int,
    agent_timeout: int,
    trajectory_file: Path,
) -> int:
    instruction = build_agent_instruction(grader_url, max_submissions)
    command = " ".join(
        [
            f"GRADER_URL={shlex.quote(grader_url)}",
            "opencode",
            "run",
            "--model",
            shlex.quote(model),
            "--dir",
            shlex.quote(REMOTE_TASK_DIR),
            "--dangerously-skip-permissions",
            shlex.quote(instruction),
        ]
    )

    print("Starting OpenCode agent...")
    process: ContainerProcess = sandbox.exec(
        "bash",
        "-lc",
        command,
        secrets=[
            modal.Secret.from_name(openai_secret_name, required_keys=["OPENAI_API_KEY"])
        ],
        timeout=agent_timeout,
        pty=True,
        workdir=REMOTE_TASK_DIR,
    )
    process.wait()

    stdout = process.stdout.read()
    stderr = process.stderr.read()
    trajectory_file.parent.mkdir(parents=True, exist_ok=True)
    trajectory_file.write_text(
        "\n".join(
            [
                "Command:",
                command,
                "",
                "Stdout:",
                stdout,
                "",
                "Stderr:",
                stderr,
            ]
        )
    )
    print(f"Agent trajectory saved to {trajectory_file}")

    return process.returncode


def run_smoke_test(
    sandbox: modal.Sandbox, grader_url: str, agent_timeout: int, trajectory_file: Path
) -> int:
    command = f"GRADER_URL={shlex.quote(grader_url)} python submit.py"
    print("Starting smoke test submission...")
    process: ContainerProcess = sandbox.exec(
        "bash",
        "-lc",
        command,
        timeout=agent_timeout,
        workdir=REMOTE_TASK_DIR,
    )
    process.wait()

    stdout = process.stdout.read()
    stderr = process.stderr.read()
    trajectory_file.parent.mkdir(parents=True, exist_ok=True)
    trajectory_file.write_text(
        "\n".join(
            [
                "Command:",
                command,
                "",
                "Stdout:",
                stdout,
                "",
                "Stderr:",
                stderr,
            ]
        )
    )
    print(f"Smoke test output saved to {trajectory_file}")

    return process.returncode


# ## Fetch the submitted solutions

# Once the agent process exits, the verifier still has its in-memory submission
# list. We fetch that list over HTTP and print it locally.


def fetch_submissions(grader_url: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{grader_url}/submissions", timeout=30) as response:
        return json.loads(response.read().decode())


def summarize_solution(solution: list[Any], preview_count: int = 5) -> dict[str, Any]:
    if len(solution) <= 2 * preview_count:
        return {"values": solution}

    return {
        "first": solution[:preview_count],
        "last": solution[-preview_count:],
        "omitted": len(solution) - 2 * preview_count,
    }


def summarize_submission(record: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "valid": record.get("valid"),
        "c1": record.get("c1"),
        "score": record.get("score"),
        "length": record.get("length"),
        "reported": record.get("reported"),
        "message": record.get("message"),
    }
    if error := record.get("error"):
        summary["error"] = error
    if solution := record.get("solution"):
        summary["solution_preview"] = summarize_solution(solution)
    return summary


def print_submission_summary(payload: dict[str, Any], max_results: int = 3) -> None:
    submissions = [
        item for item in payload.get("submissions", []) if item.get("valid")
    ][:max_results]
    if not submissions:
        print("\nNo valid solution was submitted.")
        return

    print("\nBest submitted solutions:\n")
    print(json.dumps([summarize_submission(item) for item in submissions], indent=2))


def save_best_solution(payload: dict[str, Any], best_solution_file: Path) -> None:
    best = payload.get("best")
    if not best:
        return

    best_solution_file.parent.mkdir(parents=True, exist_ok=True)
    best_solution_file.write_text(json.dumps(best, indent=2) + "\n")
    print(f"\nBest solution saved to {best_solution_file}")


# ## Run the workflow


def main(
    app_name: str,
    model: str,
    openai_secret_name: str,
    timeout: int,
    agent_timeout: int,
    max_submissions: int,
    keep_sandboxes: bool,
    smoke_test: bool,
    trajectory_file: Path,
    best_solution_file: Path,
) -> None:
    app = modal.App.lookup(app_name, create_if_missing=True)

    grader: modal.Sandbox | None = None
    agent: modal.Sandbox | None = None

    try:
        grader_image = define_grader_image()
        agent_image = define_agent_image()

        print("Starting verifier sidecar...")
        grader, grader_url = start_grader_sandbox(app, grader_image, timeout)
        print(f"Verifier sidecar: {grader.object_id}")
        print(f"Verifier URL: {grader_url}")

        agent = start_agent_sandbox(app, agent_image, timeout)
        print(f"Agent Sandbox: {agent.object_id}")

        if smoke_test:
            returncode = run_smoke_test(
                agent, grader_url, agent_timeout, trajectory_file
            )
        else:
            returncode = run_agent(
                agent,
                grader_url,
                model,
                openai_secret_name,
                max_submissions,
                agent_timeout,
                trajectory_file,
            )
        if returncode != 0:
            process_name = "Smoke test" if smoke_test else "OpenCode"
            print(f"{process_name} exited with code {returncode}.")

        submissions = fetch_submissions(grader_url)
        print_submission_summary(submissions)
        save_best_solution(submissions, best_solution_file)
        if smoke_test and not submissions.get("best"):
            raise RuntimeError("Smoke test did not submit a valid solution")
    finally:
        if keep_sandboxes:
            if grader:
                print(f"Leaving verifier sidecar running: {grader.object_id}")
            if agent:
                print(f"Leaving agent Sandbox running: {agent.object_id}")
            return

        for sandbox in (agent, grader):
            if sandbox:
                sandbox.terminate()


# ## Command-line options


def parse_timeout(timeout_str: str) -> int:
    if timeout_str.endswith("h"):
        minutes = int(timeout_str[:-1]) * 60
    elif timeout_str.endswith("m"):
        minutes = int(timeout_str[:-1])
    else:
        minutes = int(timeout_str) * 60

    if minutes < 1:
        raise argparse.ArgumentTypeError("Timeout must be at least 1 minute")
    if minutes > 24 * 60:
        raise argparse.ArgumentTypeError("Timeout cannot exceed 24 hours")

    return minutes * MINUTES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run OpenCode against a verifier sidecar on Modal"
    )
    parser.add_argument(
        "--app-name",
        default="example-opencode-autoresearch-with-eval",
        help="Modal App name. Default: example-opencode-autoresearch-with-eval",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenCode model in provider/model format. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--openai-secret",
        default=DEFAULT_OPENAI_SECRET,
        help=f"Modal Secret containing OPENAI_API_KEY. Default: {DEFAULT_OPENAI_SECRET}",
    )
    parser.add_argument(
        "--timeout",
        default="45m",
        help="Sandbox timeout (e.g. 2h, 90m). No suffix -> hours. Default: 45m",
    )
    parser.add_argument(
        "--agent-timeout",
        default="30m",
        help="OpenCode process timeout. No suffix -> hours. Default: 30m",
    )
    parser.add_argument(
        "--max-submissions",
        type=int,
        default=5,
        help="Maximum submissions to ask the agent to try. Default: 3",
    )
    parser.add_argument(
        "--keep-sandboxes",
        action="store_true",
        help="Leave both Sandboxes running for inspection after the agent exits.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast CI smoke test without invoking OpenCode or external LLM APIs.",
    )
    parser.add_argument(
        "--trajectory-file",
        type=Path,
        default=DEFAULT_TRAJECTORY_FILE,
        help=f"Where to save the full agent stdout/stderr. Default: {DEFAULT_TRAJECTORY_FILE}",
    )
    parser.add_argument(
        "--best-solution-file",
        type=Path,
        default=DEFAULT_BEST_SOLUTION_FILE,
        help=f"Where to save the full best solution JSON. Default: {DEFAULT_BEST_SOLUTION_FILE}",
    )

    args = parser.parse_args()

    main(
        app_name=args.app_name,
        model=args.model,
        openai_secret_name=args.openai_secret,
        timeout=parse_timeout(args.timeout),
        agent_timeout=parse_timeout(args.agent_timeout),
        max_submissions=args.max_submissions,
        keep_sandboxes=args.keep_sandboxes,
        smoke_test=args.smoke_test,
        trajectory_file=args.trajectory_file,
        best_solution_file=args.best_solution_file,
    )
