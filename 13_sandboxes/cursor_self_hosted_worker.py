# ---
# cmd: ["python", "13_sandboxes/cursor_self_hosted_worker.py"]
# pytest: false
# ---

# # Run Cursor self-hosted cloud agent workers in Modal Sandboxes

# This example demonstrates how to run [Cursor self-hosted cloud agent
# workers](https://cursor.com/docs/cloud-agent/self-hosted) inside Modal
# [Sandboxes](https://modal.com/docs/guide/sandboxes).
#
# In Cursor's architecture, each agent session gets its own dedicated
# worker process, started with `agent worker start`. Each worker connects
# outbound over HTTPS, so they do not need inbound ports, firewall
# changes, or VPN tunnels.

import argparse

import modal

MINUTES = 60
DEFAULT_TIMEOUT_HOURS = 12
DEFAULT_GITHUB_REPO = "modal-labs/modal-examples"


# ## Build an Image with Cursor Agent installed

# We install the official Cursor Agent CLI using Cursor's install script.
# The installer creates `~/.local/bin/agent` so we add that directory to
# `PATH`.

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git")
    .env({"PATH": "/root/.local/bin:${PATH}"})
    .run_commands("curl -fsSL https://cursor.com/install | bash")
)


# ## Grant Cursor and GitHub access

# Cursor authentication should come from a Modal
# [Secret](https://modal.com/docs/guide/secrets) containing `CURSOR_API_KEY`.
# Create one in the Modal Dashboard and use `cursor-worker-secret` as the
# default name, or pass `--cursor-secret` to use a different Secret name.
#
# For private GitHub repositories, pass a personal access token with
# `--github-token`. The script converts that token into an ephemeral Modal
# Secret for the Sandboxes it launches.
#
# Each worker serves one repository checkout, specified using
# `--github-repo` in `owner/repo` format. The repository is cloned into
# `/code/<owner>/<repo>`.


# ## Start a worker inside a Sandbox

# The worker itself is just a process, started with `agent worker start`.
# If we provide a GitHub token, we only use it for the one repository this
# worker serves. We clone via an authenticated HTTPS URL and update that
# checkout's `origin` remote.

WORKER_START_SCRIPT = r"""
set -euo pipefail

repo="${1}"
repo_dir="${2}"
clone_url="https://github.com/${repo}.git"

if [ -n "${GH_TOKEN:-}" ]; then
  clone_url="https://oauth2:${GH_TOKEN}@github.com/${repo}.git"
  export GIT_TERMINAL_PROMPT=0
fi

mkdir -p "$(dirname "${repo_dir}")"
git clone --quiet --depth 1 "${clone_url}" "${repo_dir}"
if [ -n "${GH_TOKEN:-}" ]; then
  git -C "${repo_dir}" remote set-url origin "${clone_url}"
fi
cd "${repo_dir}"

exec /root/.local/bin/agent worker start
""".strip()


def create_worker_sandbox(
    *,
    app: modal.App,
    timeout: int,
    cursor_secret_name: str,
    github_repo: str,
    github_repo_dir: str,
    github_token: str | None,
) -> modal.Sandbox:
    cursor_secret = modal.Secret.from_name(
        cursor_secret_name,
        required_keys=["CURSOR_API_KEY"],
    )

    secrets = [cursor_secret]
    if github_token:
        secrets.append(modal.Secret.from_dict({"GH_TOKEN": github_token}))

    worker_start_cmd = [
        "bash",
        "-lc",
        WORKER_START_SCRIPT,
        "worker-start",
        github_repo,
        github_repo_dir,
    ]

    return modal.Sandbox.create(
        *worker_start_cmd,
        app=app,
        image=image,
        secrets=secrets,
        timeout=timeout,
    )


# ## Putting it all together


def main(
    timeout: int,
    app_name: str,
    cursor_secret_name: str,
    github_token: str | None,
    github_repo: str,
    num_workers: int,
):
    app = modal.App.lookup(app_name, create_if_missing=True)
    github_repo_dir = github_repo_to_workdir(github_repo)

    print(f"\nStarting {num_workers} Cursor Agent workers for {github_repo}...\n")
    with modal.enable_output():
        for i in range(num_workers):
            sandbox = create_worker_sandbox(
                app=app,
                timeout=timeout,
                cursor_secret_name=cursor_secret_name,
                github_token=github_token,
                github_repo=github_repo,
                github_repo_dir=github_repo_dir,
            )
            print(f"Worker {i}: sandbox_id={sandbox.object_id}")

    print(f"\nModal dashboard: {app.get_dashboard_url()}")
    print("Cursor dashboard: https://cursor.com/dashboard/cloud-agents")


# ## Command-line options

# This script supports configuration via command-line arguments.
# Run with `--help` to see all options.

# To grant the agent the same GitHub permissions you have, you can pass a GitHub personal access token.
# If you use the `gh` CLI, you can use shell command substitution to pass your current auth:

# ```bash
#     python 13_sandboxes/cursor_self_hosted_worker.py --github-token $(gh auth token) ...
# ```


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


def parse_num_workers(value: str) -> int:
    num_workers = int(value)
    if num_workers < 1:
        raise argparse.ArgumentTypeError("--num-workers must be at least 1")
    return num_workers


def github_repo_to_workdir(github_repo: str) -> str:
    parts = [part for part in github_repo.split("/") if part]
    if len(parts) == 2:
        owner, repo = parts
        repo = repo.removesuffix(".git")
        if owner and repo:
            return f"/code/{owner}/{repo}"

    raise argparse.ArgumentTypeError("--github-repo must be in owner/repo format")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch Cursor self-hosted workers in Modal Sandboxes"
    )
    parser.add_argument(
        "--timeout",
        type=str,
        default=str(DEFAULT_TIMEOUT_HOURS),
        help=(
            "Worker timeout (e.g. 2h, 90m). "
            f"No suffix -> hours. Default: {DEFAULT_TIMEOUT_HOURS}"
        ),
    )
    parser.add_argument(
        "--app-name",
        type=str,
        default="example-cursor-self-hosted-worker",
        help="Modal app name. Default: example-cursor-self-hosted-worker",
    )
    parser.add_argument(
        "--cursor-secret",
        dest="cursor_secret_name",
        type=str,
        default="cursor-worker-secret",
        help="Modal Secret containing CURSOR_API_KEY. Default: cursor-worker-secret",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=None,
        help="GitHub PAT for private repositories. Tip: use $(gh auth token)",
    )
    parser.add_argument(
        "--github-repo",
        type=str,
        default=DEFAULT_GITHUB_REPO,
        help=(
            "GitHub repository to clone before starting the worker, "
            f"in owner/repo format. Default: {DEFAULT_GITHUB_REPO}"
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=parse_num_workers,
        default=1,
        help="Number of worker Sandboxes to launch. Default: 1",
    )

    args = parser.parse_args()

    main(
        parse_timeout(args.timeout),
        args.app_name,
        args.cursor_secret_name,
        args.github_token,
        args.github_repo,
        args.num_workers,
    )
