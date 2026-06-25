# ---
# cmd: ["modal", "run", "13_sandboxes/computer_use_vnc.py"]
# pytest: false
# ---

# # Watch Browser Use drive Chromium over VNC

# This example creates a Modal [Endpoint](https://modal.com/docs/guide/endpoints),
# then runs [Browser Use](https://docs.browser-use.com/) in a Modal
# [Sandbox](https://modal.com/docs/guide/sandbox) with a visible Chromium
# desktop. The terminal prints each browser step in addition to a noVNC URL that
# lets you watch Chromium click around while the agent works.
#
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

ENDPOINT_MODEL = "Qwen/Qwen3.6-27B-FP8"
ENDPOINT_NAME = "example-computer-use-vnc"
ENDPOINT_WARMUP_TIME = 5 * MINUTES

DEFAULT_TASK = "Read through a few subpages of the Modal docs: https://modal.com/docs. Then, tell me what Modal does."

VNC_PORT = 6080
VNC_WARMUP_TIME = 1 * MINUTES
SANDBOX_TIMEOUT = 60 * MINUTES  # up to 24 hours


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("novnc", "websockify", "x11vnc", "xvfb")
    .uv_pip_install("browser-use==0.13.1", "playwright==1.60.0")
    .run_commands("playwright install --with-deps chromium")
)


def is_server_up(url: str) -> bool:
    try:
        return urllib.request.urlopen(url).getcode() == 200
    except Exception:
        return False


# ## Pinging the Endpoint
#
# When we create an Endpoint, it's considered "live" once it's finished provisioning.
# However, it isn't necessarily ready to serve requests. The containers may have
# scaled down to zero and need to be warmed. We also poll `/health` until it returns
# HTTP 200, the same pattern we use for noVNC above.


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


@app.local_entrypoint()
def main(task: str = DEFAULT_TASK):
    if is_endpoint_live():
        print(f"Using existing endpoint {ENDPOINT_NAME!r}.")
    else:
        subprocess.run(
            [
                "modal",
                "endpoint",
                "create",
                "--name",
                ENDPOINT_NAME,
                "--model",
                ENDPOINT_MODEL,
                "--unauthenticated",
            ],
            check=False,
        )

    workspace = modal.Workspace.from_context()
    workspace.hydrate()
    environment = os.environ.get("MODAL_ENVIRONMENT", "examples")
    workspace_prefix = (
        workspace.name
        if environment in ("", "main")
        else f"{workspace.name}-{environment}"
    )
    base_url = (
        f"https://{workspace_prefix}--ep-{ENDPOINT_NAME}-server.us-west.modal.direct"
    )
    print(f"Endpoint URL: {base_url}")

    vnc_boot_command = textwrap.dedent(
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

    agent_command = textwrap.dedent(
        f"""
        import asyncio
        import json

        from browser_use import Agent, BrowserProfile, ChatOpenAI, Tools

        def _progress(msg: str) -> None:
            print(msg, flush=True)

        async def on_step_end(agent) -> None:
            _progress(f"--- Step {{agent.state.n_steps}} ---")

            urls = agent.history.urls()
            if urls and urls[-1]:
                _progress(f"URL: {{urls[-1]}}")

            thoughts = agent.history.model_thoughts()
            if thoughts:
                latest = thoughts[-1]
                memory = getattr(latest, "memory", None)
                next_goal = getattr(latest, "next_goal", None)
                if memory:
                    _progress(f"Memory: {{memory}}")
                if next_goal:
                    _progress(f"Next goal: {{next_goal}}")

            actions = agent.history.model_actions()
            if actions:
                latest_action = actions[-1]
                for name, params in latest_action.items():
                    if name == "interacted_element":
                        continue
                    _progress(f"Action: {{name}} {{json.dumps(params, default=str)}}")
                    break

        async def main() -> None:
            llm = ChatOpenAI(
                model={ENDPOINT_MODEL!r},
                api_key="unused",
                base_url={f"{base_url}/v1"!r},
                reasoning_effort="none",
                reasoning_models=[{ENDPOINT_MODEL!r}],
                timeout={3 * MINUTES!r},
            )

            agent = Agent(
                task={task!r},
                llm=llm,
                tools=Tools(),
                browser_profile=BrowserProfile(
                    headless=False,
                    window_size={{"width": 1280, "height": 720}},
                    chromium_sandbox=False,
                    args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
                ),
                use_thinking=False,
                llm_timeout={3 * MINUTES!r},
            )
            history = await agent.run(on_step_end=on_step_end)
            _progress("--- Agent finished ---")
            _progress(history.final_result())

        asyncio.run(main())
        """
    ).strip()

    sandbox = None
    try:
        with modal.enable_output():
            sandbox = modal.Sandbox.create(
                "bash",
                "-lc",
                vnc_boot_command,
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
            agent_command,
            env={"DISPLAY": ":99", "PYTHONUNBUFFERED": "1"},
            bufsize=1,
        )
        for line in agent_process.stdout:
            print(line, end="")
        agent_process.wait()
        stderr = agent_process.stderr.read()
        if stderr:
            print(stderr, end="")
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        if sandbox is not None:
            sandbox.terminate()
            print("Sandbox terminated.")
