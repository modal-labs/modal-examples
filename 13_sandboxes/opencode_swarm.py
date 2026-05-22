# ---
# cmd: ["python", "13_sandboxes/opencode_swarm.py"]
# pytest: false
# ---

# # Replicate the Sailboxes experiment on Modal

# [Sail Research](https://www.sailresearch.com/news/introducing-sailboxes-persistent-sandboxes)
# demonstrated a swarm of four coding agents building a
# [wire-compatible Redis clone in Rust](https://github.com/sailresearchco/sails-redis),
# running for over 100 cumulative hours.

# This example shows how to replicate their setup on Modal:
# a swarm of [OpenCode](https://opencode.ai/docs) Sandboxes,
# each powered by a self-hosted [GLM-5](https://huggingface.co/zai-org/GLM-5-FP8) inference server.

# ## Architecture

# 1. **GLM-5 inference**: served via SGLang on 8×H200 GPUs using the
#    [very large models example](https://modal.com/docs/examples/very_large_models).
# 2. **Agent Sandboxes**: four OpenCode instances, each with a Rust toolchain
#    and a clone of the target repo, configured to call the GLM-5 endpoint.
# 3. **Persistence**: [Sandbox Snapshots](https://modal.com/docs/guide/sandbox-snapshots)
#    let you save and resume work across the 24-hour Sandbox lifetime limit.

# ## Prerequisites

# Deploy GLM-5 before running this script.
# In `06_gpu_and_ml/llm-serving/very_large_models.py`, change the model constants:

# ```python
# REPO_ID = "zai-org/GLM-5-FP8"  # was GLM-4.7-FP8
# GPU_COUNT = 8                    # was 4; GLM-5 needs ~740 GB in FP8
# ```

# Then deploy with the GLM-5 config:

# ```bash
# APP_LOCAL_CONFIG_PATH=06_gpu_and_ml/llm-serving/config_glm5.yaml \
#   modal deploy 06_gpu_and_ml/llm-serving/very_large_models.py
# ```

# Grab the endpoint URL from the deploy output and pass it here:

# ```bash
# python 13_sandboxes/opencode_swarm.py --endpoint https://your-glm5-endpoint.modal.run
# ```

# ## Set up the Sandbox Image

import argparse
import json
import secrets

import modal

MINUTES = 60
HOURS = 60 * MINUTES
OPENCODE_PORT = 4096
NUM_AGENTS = 4
DEFAULT_GITHUB_REPO = "sailresearchco/sails-redis"


# Each Sandbox needs OpenCode, a Rust toolchain, and basic development tools.


def define_agent_image() -> modal.Image:
    return (
        modal.Image.debian_slim()
        .apt_install("curl", "git", "build-essential", "pkg-config")
        # Install Rust
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"
        )
        .env({"PATH": "/root/.cargo/bin:${PATH}"})
        # Install OpenCode
        .run_commands("curl -fsSL https://opencode.ai/install | bash")
        .env({"PATH": "/root/.opencode/bin:${PATH}"})
    )


# ## Clone the target repository

# By default, we clone
# [sails-redis](https://github.com/sailresearchco/sails-redis),
# the same Redis-in-Rust project from the original Sailboxes experiment.


def clone_repo(image: modal.Image, repo: str, ref: str) -> modal.Image:
    clone_cmd = (
        f"GIT_TERMINAL_PROMPT=0 git clone --quiet --depth 1 --branch {ref} "
        f"https://github.com/{repo}.git /root/code"
    )
    print(f"🏖️  Cloning {repo}@{ref}")
    return image.run_commands(
        "git config --global advice.detachedHead false",
        clone_cmd,
        force_build=True,
    )


# ## Configure OpenCode to use GLM-5

# We write an `opencode.json` that points OpenCode at our self-hosted
# GLM-5 endpoint using the
# [custom provider](https://opencode.ai/docs/providers#custom-provider) format.


def write_opencode_config(image: modal.Image, endpoint_url: str) -> modal.Image:
    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "modal-glm5": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Modal GLM-5",
                "options": {"baseURL": endpoint_url.rstrip("/") + "/v1"},
                "models": {
                    "zai-org/GLM-5-FP8": {
                        "name": "GLM-5 FP8",
                        "limit": {"context": 131072, "output": 32768},
                    }
                },
            }
        },
        "model": {"provider": "modal-glm5", "model": "zai-org/GLM-5-FP8"},
    }

    config_json = json.dumps(config, indent=2)

    return image.run_commands(
        "mkdir -p /root/.config/opencode",
        f"cat > /root/.config/opencode/opencode.json << 'ENDOFCONFIG'\n{config_json}\nENDOFCONFIG",
    )


# ## Create the agent Sandboxes

# Each Sandbox runs OpenCode in
# [server mode](https://opencode.ai/docs/server),
# exposing a Web UI and TUI attachment endpoint.
# We generate a random password for each agent.


def create_agent_sandbox(
    image: modal.Image,
    app: modal.App,
    agent_id: int,
    timeout: int,
    snapshot_image: modal.Image | None = None,
) -> modal.Sandbox:
    password = secrets.token_urlsafe(16)

    resolved_image = snapshot_image if snapshot_image is not None else image

    print(f"🏖️  Creating agent {agent_id}")

    with modal.enable_output():
        sb = modal.Sandbox.create(
            "opencode",
            "serve",
            "--hostname=0.0.0.0",
            f"--port={OPENCODE_PORT}",
            "--log-level=DEBUG",
            "--print-logs",
            encrypted_ports=[OPENCODE_PORT],
            secrets=[modal.Secret.from_dict({"OPENCODE_SERVER_PASSWORD": password})],
            timeout=timeout,
            image=resolved_image,
            app=app,
            workdir="/root/code",
        )

    tunnel = sb.tunnels()[OPENCODE_PORT]
    print(
        f"🏖️  Agent {agent_id} ready:",
        f"\tSandbox: modal shell {sb.object_id}",
        f"\tWeb UI:  {tunnel.url}",
        "\t  user:  opencode",
        f"\t  pass:  {password}",
        f"\tTUI:     OPENCODE_SERVER_PASSWORD={password} opencode attach {tunnel.url}",
        sep="\n",
    )

    return sb


# ## Snapshot a Sandbox for later resumption

# Before a Sandbox approaches its timeout, snapshot the filesystem
# so you can resume in a new Sandbox without losing progress.
# See [Sandbox Snapshots](https://modal.com/docs/guide/sandbox-snapshots).


def snapshot_agent(sandbox: modal.Sandbox, agent_id: int) -> modal.Image:
    print(f"🏖️  Snapshotting agent {agent_id} ({sandbox.object_id})...")
    image = sandbox.snapshot_filesystem()
    print(f"🏖️  Snapshot saved: {image.object_id}")
    return image


# ## Putting it all together

# Launch the full swarm: build the Image once, then fan out into
# `NUM_AGENTS` parallel Sandboxes.


def main(
    endpoint_url: str,
    num_agents: int,
    timeout: int,
    app_name: str,
    github_repo: str,
    github_ref: str,
    resume_snapshots: list[str] | None = None,
):
    app = modal.App.lookup(app_name, create_if_missing=True)

    image = define_agent_image()
    image = clone_repo(image, github_repo, github_ref)
    image = write_opencode_config(image, endpoint_url)

    sandboxes = []
    for i in range(num_agents):
        snap = None
        if resume_snapshots and i < len(resume_snapshots):
            snap = modal.Image.from_id(resume_snapshots[i])
            print(f"🏖️  Resuming agent {i} from snapshot {resume_snapshots[i]}")

        sb = create_agent_sandbox(image, app, i, timeout, snapshot_image=snap)
        sandboxes.append(sb)

    print(
        f"\n🏖️  Swarm launched: {len(sandboxes)} agents working on {github_repo}",
        "🏖️  Sandboxes will run for up to "
        f"{timeout // HOURS}h {(timeout % HOURS) // MINUTES}m.",
        "🏖️  To snapshot before timeout, run:",
        "      modal sandbox snapshot <sandbox-id>",
        "🏖️  To resume from snapshots later, pass --resume with image IDs.",
        sep="\n",
    )


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
        description="Launch an OpenCode agent swarm on Modal"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        required=True,
        help="GLM-5 inference endpoint URL (from `modal deploy very_large_models.py`)",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=NUM_AGENTS,
        help=f"Number of agent Sandboxes to launch. Default: {NUM_AGENTS}",
    )
    parser.add_argument(
        "--timeout",
        type=str,
        default="24",
        help="Per-Sandbox timeout (e.g. 2h, 90m). No suffix -> hours. Default: 24",
    )
    parser.add_argument(
        "--app-name",
        type=str,
        default="example-opencode-swarm",
        help="Modal App name. Default: example-opencode-swarm",
    )
    parser.add_argument(
        "--github-repo",
        type=str,
        default=DEFAULT_GITHUB_REPO,
        help=f"GitHub repo to clone. Default: {DEFAULT_GITHUB_REPO}",
    )
    parser.add_argument(
        "--github-ref",
        type=str,
        default="main",
        help="Git ref to check out. Default: main",
    )
    parser.add_argument(
        "--resume",
        nargs="*",
        metavar="IMAGE_ID",
        help="Resume agents from Filesystem Snapshot Image IDs (e.g. im-abc123)",
    )

    args = parser.parse_args()

    main(
        endpoint_url=args.endpoint,
        num_agents=args.num_agents,
        timeout=parse_timeout(args.timeout),
        app_name=args.app_name,
        github_repo=args.github_repo,
        github_ref=args.github_ref,
        resume_snapshots=args.resume,
    )
