# ---
# cmd: ["modal", "serve", "13_sandboxes/computer_use_vnc.py"]
# pytest: false
# ---

# # Watch Browser Use drive Chromium over VNC

# Computer-use agents are LLMs that can interact with a web browser in a loop.
# Rather than calling a fixed set of APIs, they look at a rendered page or screen,
# decide what to click or type next, take that action, and look again.
#
# This example builds one with [Browser Use](https://docs.browser-use.com/).
# An open-weights model served from a Modal
# [Endpoint](https://modal.com/docs/guide/endpoints) powers the agent. The agent
# drives Chromium inside a Modal
# [VM Sandbox](https://modal.com/docs/guide/vm-sandboxes),
# while a small web UI embeds a noVNC desktop so you can watch it work.

# ## Run the example
#
# ```bash
# modal serve 13_sandboxes/computer_use_vnc.py
# ```
#

import asyncio
import json
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

import fastapi
import modal
from fastapi.responses import HTMLResponse

app = modal.App("example-computer-use-vnc")
MINUTES = 60
# We could point Browser Use at a hosted provider like OpenAI or Anthropic using
# your API key. For our purposes, however, we serve an open-weights model
# ourselves via a Modal [Endpoint](https://modal.com/docs/guide/endpoints).
# It takes one command to create an OpenAI-compatible server for the agent to
# call. No external API key is needed, and the whole demo runs on Modal.

ENDPOINT_MODEL = "Qwen/Qwen3.6-27B-FP8"
ENDPOINT_NAME = "example-computer-use-vnc"
ENDPOINT_ROUTING_REGION = "us-west"
ENDPOINT_WARMUP_TIME = 5 * MINUTES
endpoint_server = modal.Server.from_name(f"ep-{ENDPOINT_NAME}", "Server")

VNC_PORT = 6080
SESSION_START_TIMEOUT = 2 * MINUTES
SANDBOX_TIMEOUT = 60 * MINUTES

PAGE_PATH = Path(__file__).parent / "computer_use_vnc.html"
PAGE_REMOTE = "/root/computer_use_vnc.html"
RESULT_PREFIX = "__BROWSER_USE_RESULT__="
DESKTOP_READY_PATH = "/tmp/desktop_ready"

# ## Set up a shareable virtual desktop

# By default, Browser Use launches Chromium headless. However, we want to watch the
# browser in real time. In each Sandbox, Xvfb provides a virtual display,
# x11vnc serves it over VNC, and websockify bridges the stream into the noVNC
# page that the UI embeds.

base_image = modal.Image.debian_slim(python_version="3.12")
web_image = base_image.uv_pip_install("fastapi[standard]==0.139.2").add_local_file(
    PAGE_PATH, remote_path=PAGE_REMOTE
)
sandbox_image = (
    base_image.apt_install("novnc", "websockify", "x11vnc", "xvfb")
    .uv_pip_install("browser-use==0.13.6", "playwright==1.61.0")
    .run_commands("playwright install --with-deps chromium")
)

SANDBOX_COMMAND = textwrap.dedent(
    """
    set -euo pipefail
    export DISPLAY=:99
    Xvfb :99 -screen 0 1280x720x24 >/tmp/xvfb.log 2>&1 &
    sleep 1
    x11vnc -display :99 -forever -shared -nopw -listen 0.0.0.0 -rfbport 5900 -xkb >/tmp/x11vnc.log 2>&1 &
    websockify --web=/usr/share/novnc/ 6080 localhost:5900 >/tmp/websockify.log 2>&1 &
    exec python -c "$AGENT_SCRIPT"
    """
).strip()

# ## Driving the browser with an agent loop

# The agent loop is the core of the agent. At each step, it looks at the current
# page, picks an action such as click, type, or navigate, and executes it with
# Browser Use. The loop repeats until the model decides the task is done.

AGENT_SCRIPT = textwrap.dedent(
    """
    import asyncio
    import json
    import os
    import time
    import urllib.parse
    import urllib.request
    from pathlib import Path

    from browser_use import Agent, Browser, ChatOpenAI, Tools

    model = os.environ["ENDPOINT_MODEL"]
    base_url = os.environ["ENDPOINT_BASE_URL"]
    start_page = "data:text/html," + urllib.parse.quote(
        "<body style='margin:0;background:#222;color:#ddd;font:28px system-ui;"
        "display:grid;place-items:center;height:100vh'>Starting desktop...</body>"
    )


    def wait_for_endpoint() -> None:
        deadline = time.monotonic() + int(os.environ["ENDPOINT_WARMUP_TIME"])
        while True:
            try:
                urllib.request.urlopen(f"{base_url}/health", timeout=5).close()
                return
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for the model Endpoint.")
            time.sleep(1)


    async def main() -> None:
        browser = Browser(
            headless=False,
            window_size={"width": 1280, "height": 720},
            chromium_sandbox=False,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
        )
        await browser.start()
        await browser.navigate_to(start_page)
        Path(os.environ["DESKTOP_READY_PATH"]).write_text("1", encoding="utf-8")
        wait_for_endpoint()
        llm = ChatOpenAI(
            model=model,
            api_key="unused",
            base_url=f"{base_url}/v1",
            reasoning_effort="none",
            reasoning_models=[model],
            timeout=3 * 60,
        )
        agent = Agent(
            task=os.environ["AGENT_TASK"],
            llm=llm,
            tools=Tools(),
            browser=browser,
            use_thinking=False,
            llm_timeout=3 * 60,
        )
        history = await agent.run()
        result = history.final_result() or "Agent stopped without a final result."
        print(os.environ["RESULT_PREFIX"] + json.dumps(result), flush=True)


    asyncio.run(main())
    """
).strip()

# ## Creating the shared Endpoint

# When you serve the web UI, we create a shared Endpoint that will be used across requests.
# The Endpoint can take time to become ready because its containers scale to zero.
# Startup waits in two places:
# 1. The Sandbox waits until the endpoint is ready before starting the agent.
# 2. `start_session` waits for the endpoint and the noVNC HTTP server to be ready before returning `watch_url`.

if modal.is_local():
    command = [sys.executable, "-m", "modal", "endpoint"]
    endpoints = json.loads(
        subprocess.check_output([*command, "list", "--json"], text=True)
    )
    if not any(endpoint["name"] == ENDPOINT_NAME for endpoint in endpoints):
        subprocess.run(
            [
                *command,
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
        print(f"Created Endpoint {ENDPOINT_NAME!r}.")
    else:
        print(f"Using existing Endpoint {ENDPOINT_NAME!r}.")


def is_server_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


# ## Running the agent

# A request from the UI creates one Sandbox for one task. Its entrypoint starts
# the virtual desktop, paints Chromium, then runs Browser Use. `start_session`
# waits until the desktop-ready marker exists and the noVNC page responds, then
# returns the Sandbox ID and watch URL.
#
# The browser embeds that URL and polls the status route with the ID. When the
# agent exits, the Sandbox terminates and the status route returns its final
# result. A failed startup terminates the Sandbox before returning the error.


@app.function(image=web_image, timeout=SESSION_START_TIMEOUT + 30)
async def start_session(task: str):
    sandbox = None
    try:
        endpoint_url = await endpoint_server.get_url.aio()
        if endpoint_url is None:
            raise RuntimeError(f"Endpoint {ENDPOINT_NAME!r} has no URL.")
        deadline = time.monotonic() + SESSION_START_TIMEOUT
        sandbox = await modal.Sandbox.create.aio(
            "bash",
            "-lc",
            SANDBOX_COMMAND,
            app=app,
            image=sandbox_image,
            experimental_options={"vm_runtime": True},
            env={
                "AGENT_SCRIPT": AGENT_SCRIPT,
                "AGENT_TASK": task,
                "DESKTOP_READY_PATH": DESKTOP_READY_PATH,
                "ENDPOINT_BASE_URL": endpoint_url,
                "ENDPOINT_MODEL": ENDPOINT_MODEL,
                "ENDPOINT_WARMUP_TIME": str(ENDPOINT_WARMUP_TIME),
                "RESULT_PREFIX": RESULT_PREFIX,
            },
            encrypted_ports=[VNC_PORT],
            timeout=SANDBOX_TIMEOUT,
            readiness_probe=modal.Probe.with_exec("test", "-f", DESKTOP_READY_PATH),
        )
        remaining = max(1, int(deadline - time.monotonic()))
        await sandbox.wait_until_ready.aio(timeout=remaining)
        remaining = max(1, int(deadline - time.monotonic()))
        tunnel = (await sandbox.tunnels.aio(timeout=remaining))[VNC_PORT]
        watch_url = (
            f"{tunnel.url.rstrip('/')}/vnc.html?autoconnect=1&resize=scale&reconnect=1"
        )
        while not is_server_up(watch_url):
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for noVNC.")
            await asyncio.sleep(1)
        return {"sandbox_id": sandbox.object_id, "watch_url": watch_url}
    except Exception:
        if sandbox is not None:
            await sandbox.terminate.aio()
        raise
    finally:
        if sandbox is not None:
            await sandbox.detach.aio()


# ## Serve the web UI

web_app = fastapi.FastAPI()


@web_app.get("/")
async def index():
    return HTMLResponse(Path(PAGE_REMOTE).read_text())


@web_app.post("/api/session")
async def create_session(body: dict):
    task = str(body.get("task", "")).strip()
    if not task:
        raise fastapi.HTTPException(status_code=400, detail="Task must not be empty.")
    try:
        return await start_session.remote.aio(task)
    except Exception as exc:
        raise fastapi.HTTPException(500, f"Starting Sandbox: {exc}") from exc


@web_app.get("/api/session/{sandbox_id}")
async def session_status(sandbox_id: str):
    try:
        sandbox = await modal.Sandbox.from_id.aio(sandbox_id)
    except modal.exception.NotFoundError as exc:
        raise fastapi.HTTPException(404, "Session not found.") from exc

    try:
        returncode = await sandbox.poll.aio()
        if returncode is None:
            return {"state": "running"}
        stdout = await sandbox.stdout.read.aio()
        stderr = await sandbox.stderr.read.aio()
    finally:
        await sandbox.detach.aio()

    if returncode == 0:
        result = None
        for line in reversed(stdout.splitlines()):
            if line.startswith(RESULT_PREFIX):
                result = json.loads(line.removeprefix(RESULT_PREFIX))
                break
        if result is None:
            result = "Agent finished without a result."
        return {"state": "succeeded", "result": result}
    message = (stderr or stdout).strip()[-4000:]
    return {
        "state": "failed",
        "result": message or f"Agent exited with code {returncode}.",
    }


@app.function(image=web_image)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    return web_app


# ## Cleaning up

# Each Sandbox uses the agent process as its entrypoint, so it stops when the
# task finishes or its timeout expires. Startup failures terminate it
# immediately, and every code path detaches its local Sandbox handle.
#
# Stop `modal serve` with Ctrl-C. The shared Endpoint scales to zero when idle,
# but remains available for later prompts. Shut it down when you are done:
#
# ```bash
# modal endpoint stop example-computer-use-vnc
# ```
