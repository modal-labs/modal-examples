# ---
# cmd: ["modal", "run", "13_sandboxes/computer_use_vnc.py"]
# pytest: false
# ---

# # Watch Browser Use drive Chromium over VNC

# Computer-use agents are LLMs that can interact with a web browser in a loop.
# Rather than calling a fixed set of APIs, they look at a rendered page or screen,
# decide what to click or type next, take that action, and look again.
#
# This example builds one with [Browser Use](https://docs.browser-use.com/).
# The agent is powered by an open-weights model we serve ourselves from a Modal [Endpoint](https://modal.com/docs/guide/endpoints),
# and drives a Chromium browser hosted inside a Modal [Sandbox](https://modal.com/docs/guide/sandbox).
# We stream each step to the terminal and expose a VNC link so you can watch it work in real time.

# ## Run the example
#
# ```bash
# modal run 13_sandboxes/computer_use_vnc.py
# ```
#
# The default task asks the agent to read the Modal docs and summarize them.
# Try something else with:
#
# ```bash
# modal run 13_sandboxes/computer_use_vnc.py --task "What's the weather in Tokyo?"
# ```

import json
import os
import subprocess
import textwrap
import time
import urllib.request

import modal

app = modal.App("example-computer-use-vnc")
MINUTES = 60

# We could point Browser Use at a hosted provider like OpenAI or Anthropic using your API key.
# For our purposes, however, we serve an open-weights model ourselves via a Modal
# [Endpoint](https://modal.com/docs/guide/endpoints). It only takes one command
# to create an OpenAI-compatible server for the agent to call.
# No external API key is needed, and the whole demo runs on Modal.

ENDPOINT_MODEL = "Qwen/Qwen3.6-27B-FP8"
ENDPOINT_NAME = "example-computer-use-vnc"
ENDPOINT_ROUTING_REGION = "us-west"
ENDPOINT_WARMUP_TIME = 5 * MINUTES

DEFAULT_TASK = "Read through a few subpages of the Modal docs: https://modal.com/docs. Then, tell me what Modal does."

VNC_PORT = 6080
VNC_WARMUP_TIME = 1 * MINUTES
SANDBOX_TIMEOUT = 60 * MINUTES  # stay alive for one hour; can be up to one day

# ## Set up a shareable virtual desktop

# By default, Browser Use launches Chromium headless.
# However, we want to watch the browser in real time.
# In the Sandbox, we set up a virtual display using Xvfb, serve it over VNC using x11vnc,
# and bridge the stream into a web page using websockify and noVNC.

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("novnc", "websockify", "x11vnc", "xvfb")
    .uv_pip_install("browser-use==0.13.1", "playwright==1.60.0")
    .run_commands("playwright install --with-deps chromium")
)

VNC_BOOT_COMMAND = textwrap.dedent(
    """
    set -euo pipefail
    export DISPLAY=:99
    Xvfb :99 -screen 0 1280x720x24 &
    sleep 1
    x11vnc -display :99 -forever -shared -nopw -listen 0.0.0.0 -rfbport 5900 -xkb &
    websockify --web=/usr/share/novnc/ 6080 localhost:5900 &
    exec sleep infinity
    """
).strip()

# ## Driving the browser with an agent loop

# The agent loop is the core of the agent.
# At each step, it looks at the current page, picks an action such as click, type, or navigate,
# and executes it with Browser Use. The loop repeats until the model decides the task is done.

# By default, nothing is printed until the whole run finishes, which is no fun to watch.
# So we pass an `on_step_end` hook that fires after every step and prints the step number,
# the current URL, the model's memory and next goal, and the action it chose.

AGENT_SCRIPT = textwrap.dedent(
    """
    import asyncio
    import json
    import os

    from browser_use import Agent, BrowserProfile, ChatOpenAI, Tools

    MINUTES = 60

    model = os.environ["ENDPOINT_MODEL"]
    base_url = os.environ["ENDPOINT_BASE_URL"]
    task = os.environ["AGENT_TASK"]


    def _progress(msg: str) -> None:
        print(msg, flush=True)


    async def on_step_end(agent) -> None:
        _progress(f"--- Step {agent.state.n_steps} ---")

        urls = agent.history.urls()
        if urls and urls[-1]:
            _progress(f"URL: {urls[-1]}")

        thoughts = agent.history.model_thoughts()
        if thoughts:
            latest = thoughts[-1]
            memory = getattr(latest, "memory", None)
            next_goal = getattr(latest, "next_goal", None)
            if memory:
                _progress(f"Memory: {memory}")
            if next_goal:
                _progress(f"Next goal: {next_goal}")

        actions = agent.history.model_actions()
        if actions:
            latest_action = actions[-1]
            for name, params in latest_action.items():
                if name == "interacted_element":
                    continue
                _progress(f"Action: {name} {json.dumps(params, default=str)}")
                break


    async def main() -> None:
        llm = ChatOpenAI(
            model=model,
            api_key="unused",
            base_url=f"{base_url}/v1",
            reasoning_effort="none",
            reasoning_models=[model],
            timeout=3 * MINUTES,
        )

        agent = Agent(
            task=task,
            llm=llm,
            tools=Tools(),
            browser_profile=BrowserProfile(
                headless=False,
                window_size={"width": 1280, "height": 720},
                chromium_sandbox=False,
                args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
            ),
            use_thinking=False,
            llm_timeout=3 * MINUTES,
        )
        history = await agent.run(on_step_end=on_step_end)
        _progress("--- Agent finished ---")
        _progress(history.final_result() or "Agent stopped without a final result.")


    asyncio.run(main())
    """
).strip()

# ## Pinging the Endpoint

# When we create an Endpoint, it becomes "live" once it finishes provisioning.
# However, it isn't ready to serve requests until at least one container is up,
# since containers scale to zero. If we were to start the agent before the Endpoint is ready,
# it would produce a wall of connection errors.
# Therefore, we have two checks:
# 1. `is_endpoint_live` to tell us whether the Endpoint is provisioned (i.e., it exists).
# 2. `is_server_up` to tell us whether the Endpoint/VNC server are ready to serve requests.


def is_endpoint_live() -> bool:
    result = subprocess.run(
        ["modal", "endpoint", "list", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    for endpoint in json.loads(result.stdout or "[]"):
        if endpoint["name"] == ENDPOINT_NAME:
            return True
    return False


def is_server_up(url: str) -> bool:
    try:
        return urllib.request.urlopen(url, timeout=5).getcode() == 200
    except Exception:
        return False


# ## Running the agent

# Here, we create the Sandbox and Endpoint, wait for them to be ready, and then run the agent.
# The VNC URL will be printed to the terminal, so you can watch the browser in real time.
# While the agent is running, the terminal will also print the agent's steps and actions.
# Once the agent is finished, we clean up (see the last section below).


@app.local_entrypoint()
def main(task: str = DEFAULT_TASK):
    if not is_endpoint_live():
        subprocess.run(
            [
                "modal",
                "endpoint",
                "create",
                "--name",
                ENDPOINT_NAME,
                "--model",
                ENDPOINT_MODEL,
                "--routing-region",
                ENDPOINT_ROUTING_REGION,
                "--unauthenticated",
            ],
            check=True,
        )
    else:
        print(f"Using existing endpoint {ENDPOINT_NAME!r}.")

    sandbox = None
    try:
        workspace = modal.Workspace.from_context()
        workspace.hydrate()
        environment = os.environ.get("MODAL_ENVIRONMENT", "main")
        workspace_prefix = (
            workspace.name
            if environment in ("", "main")
            else f"{workspace.name}-{environment}"
        )
        base_url = f"https://{workspace_prefix}--ep-{ENDPOINT_NAME}-server.{ENDPOINT_ROUTING_REGION}.modal.direct"
        print(f"Endpoint URL: {base_url}")

        with modal.enable_output():
            sandbox = modal.Sandbox.create(
                "bash",
                "-lc",
                VNC_BOOT_COMMAND,
                app=app,
                image=image,
                encrypted_ports=[VNC_PORT],
                timeout=SANDBOX_TIMEOUT,
            )

        print(f"Sandbox ID: {sandbox.object_id}")

        tunnel = sandbox.tunnels()[VNC_PORT]
        deadline = time.time() + VNC_WARMUP_TIME
        while time.time() < deadline:
            if is_server_up(tunnel.url):
                watch_url = f"{tunnel.url.rstrip('/')}/vnc.html?autoconnect=1&resize=scale&reconnect=1"
                print(f"Watch the browser at: {watch_url}")
                break
            time.sleep(1)
        else:
            raise TimeoutError("Timed out waiting for noVNC.")

        print("Waiting for the endpoint to be ready...")
        deadline = time.time() + ENDPOINT_WARMUP_TIME
        while not is_server_up(f"{base_url}/health"):
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for {ENDPOINT_NAME!r}.")
            time.sleep(1)

        agent_process = sandbox.exec(
            "python",
            "-c",
            AGENT_SCRIPT,
            env={
                "DISPLAY": ":99",
                "PYTHONUNBUFFERED": "1",
                "ENDPOINT_MODEL": ENDPOINT_MODEL,
                "ENDPOINT_BASE_URL": base_url,
                "AGENT_TASK": task,
            },
            bufsize=1,
        )
        for line in agent_process.stdout:
            print(line, end="")
        returncode = agent_process.wait()
        stderr = agent_process.stderr.read()
        if stderr:
            print(stderr, end="")
        if returncode != 0:
            print(f"Agent exited with code {returncode}.")
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        try:
            if sandbox is not None:
                sandbox.terminate()
                print("Sandbox terminated.")
        finally:
            subprocess.run(
                ["modal", "endpoint", "stop", ENDPOINT_NAME, "--yes"],
                check=False,
            )


# ## Cleaning up

# After the agent finishes its task, we terminate the Sandbox and then stop the Endpoint.
# If a run is killed before cleanup can fire (SIGKILL, a closed terminal), stop the Endpoint yourself with:
#
# ```bash
# modal endpoint stop example-computer-use-vnc
# ```
