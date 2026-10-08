# ---
# cmd: ["python", "13_sandboxes/sidecar_agent.py"]
# pytest: false
# ---

# # Separate an agent from its execution environment with Sidecars
#
# Running untrusted generated code in the same container as the agent risks
# exposing credentials or letting the code interfere with the agent itself.
# [Anthropic's agent architecture article](https://www.anthropic.com/engineering/managed-agents)
# discusses some of these risks in more detail.
#
# This example shows how to separate the agent and the code it executes with
# [Sandbox Sidecars](https://modal.com/docs/guide/sandbox-sidecars). The agent
# runs in the Sidecar, while the main
# [Sandbox](https://modal.com/docs/guide/sandboxes) executes generated code and
# returns its output. Both containers run on the same host and communicate over
# a low-latency private bridge.
#
# Here, the agent is asked to debug an invalid Python file.
#
# Run the example with:
#
# ```bash
# python 13_sandboxes/sidecar_agent.py
# ```
#

import json
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlparse

import modal

app = modal.App.lookup("example-sidecar-agent", create_if_missing=True)

MINUTES = 60  # seconds

# ## Serve the model with an Endpoint
#
# The agent is powered by a dedicated Modal
# [Endpoint](https://modal.com/docs/guide/endpoints) that serves an
# open-weights model behind an OpenAI-compatible API.

ENDPOINT_NAME = "example-sidecar-agent"
ENDPOINT_MODEL = "Qwen/Qwen3.6-27B-FP8"
endpoint_server = modal.Server.from_name(f"ep-{ENDPOINT_NAME}", "Server")


def create_endpoint_if_missing() -> None:
    command = [sys.executable, "-m", "modal", "endpoint"]
    endpoints = json.loads(
        subprocess.check_output([*command, "list", "--json"], text=True)
    )
    if any(endpoint["name"] == ENDPOINT_NAME for endpoint in endpoints):
        return
    subprocess.run(
        [
            *command,
            "create",
            "--name",
            ENDPOINT_NAME,
            "--model",
            ENDPOINT_MODEL,
            "--unauthenticated",
        ],
        check=True,
    )


def get_endpoint_model_name(url: str) -> str | None:
    try:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=5) as response:
            model_name = json.load(response)["data"][0]["id"]
        return model_name if isinstance(model_name, str) and model_name else None
    except Exception:
        return None


def wait_for_endpoint() -> tuple[str, str]:
    deadline = time.monotonic() + 10 * MINUTES
    while True:
        try:
            url = endpoint_server.get_url()
        except modal.exception.NotFoundError:
            url = None
        if url and (model_name := get_endpoint_model_name(url)):
            return url, model_name
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for Endpoint {ENDPOINT_NAME!r}.")
        time.sleep(1)


# ## Define the worker and the agent
#
# The worker is an HTTP server that runs in the main Sandbox, which has no outbound
# network access. It executes generated shell commands and returns their output.

WORKER = r"""
import os
import signal
import subprocess

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field


def execute(command):
    process = subprocess.Popen(
        ["bash", "-c", command],
        cwd="/workspace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    timed_out = False
    try:
        process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    stdout, stderr = process.communicate()
    return {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
    }


server = FastAPI()


class Command(BaseModel):
    command: str = Field(min_length=1, max_length=10000)


@server.post("/execute")
def run_command(request: Command):
    return execute(request.command)


uvicorn.run(server, host="0.0.0.0", port=8080)
"""

# The agent, which runs in the Sidecar, calls the model and forwards commands
# to the worker.

AGENT = r"""
import json
import os
import urllib.request

BASE_URL = os.environ["ENDPOINT_BASE_URL"].rstrip("/")
MODEL = os.environ["ENDPOINT_MODEL"]

TOOL = {
    "type": "function",
    "function": {
        "name": "execute",
        "description": (
            "Run a shell command in /workspace. Returns stdout, stderr, exit "
            "status, and timeout status. Commands time out after ten seconds."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


def execute(command):
    request = urllib.request.Request(
        "http://main:8080/execute",
        data=json.dumps({"command": command}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def chat(messages):
    request = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(
            {
                "model": MODEL,
                "messages": messages,
                "tools": [TOOL],
                "max_tokens": 2048,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        choice = json.load(response)["choices"][0]
    if choice["finish_reason"] == "length":
        raise RuntimeError("Model reached the output limit")
    return choice["message"]


def run_agent():
    messages = [
        {
            "role": "user",
            "content": (
                "Inspect /workspace/stats.py. Use the execute tool to run average with a few "
                "nonempty lists and check whether it computes the arithmetic mean correctly. "
                "Explain any bug using the execution results you observe. "
                "Do not modify the file. Keep the final answer brief."
            ),
        }
    ]
    used_tool = False
    for _ in range(6):
        message = chat(messages)
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            if not used_tool:
                raise RuntimeError("Model did not complete a tool-using turn")
            print("[agent]:", message.get("content") or "")
            return
        for call in calls:
            if call["function"]["name"] != "execute":
                raise RuntimeError("Unknown tool")
            command = json.loads(call["function"]["arguments"])["command"]
            output = execute(command)
            used_tool = True
            print(f"[tool]: exit={output['returncode']} timed_out={output['timed_out']}")
            print(output["stdout"], end="")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(output),
                }
            )
    raise RuntimeError("Agent did not finish within six model requests")


run_agent()
"""

# ## Start the Sandbox and Sidecar
#
# We'll start the Endpoint, write the workspace files into the Sandbox, then
# starts the agent in an attached Sidecar.
#
# Note that Sidecars require a pre-built [Image](https://modal.com/docs/guide/images), so
# the agent image is built before the Sidecar starts.


def main():
    create_endpoint_if_missing()
    endpoint_url, endpoint_model_name = wait_for_endpoint()

    worker_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "fastapi[standard]==0.139.2"
    )
    agent_image = modal.Image.debian_slim(python_version="3.12")

    with modal.enable_output():
        agent_image = agent_image.build(app)
        sandbox = modal.Sandbox.create(
            "python",
            "-u",
            "-c",
            WORKER,
            app=app,
            image=worker_image,
            outbound_cidr_allowlist=[],
            timeout=10 * MINUTES,
            readiness_probe=modal.Probe.with_tcp(8080),
        )

    try:
        sandbox.wait_until_ready()
        sandbox.filesystem.make_directory("/workspace")

        FILES = {
            "stats.py": """def average(values): return sum(values) // len(values)""",
        }
        for name, source in FILES.items():
            sandbox.filesystem.write_text(source, f"/workspace/{name}")

        sidecar = sandbox._experimental_sidecars.create(
            "sleep",
            "infinity",
            name="agent",
            image=agent_image,
            env={
                "ENDPOINT_BASE_URL": endpoint_url,
                "ENDPOINT_MODEL": endpoint_model_name,
            },
            outbound_domain_allowlist=[urlparse(endpoint_url).hostname],
        )

        check = sandbox.exec(
            "python",
            "-c",
            "import os\n"
            "from pathlib import Path\n"
            "assert 'ENDPOINT_BASE_URL' not in os.environ\n",
        )
        if check.wait() != 0:
            raise RuntimeError(check.stderr.read())

        process = sidecar.exec("python", "-u", "-c", AGENT)
        for line in process.stdout:
            print(line, end="")
        if process.wait() != 0:
            raise RuntimeError(process.stderr.read())
    finally:
        sandbox.terminate(wait=True)


if __name__ == "__main__":
    main()
