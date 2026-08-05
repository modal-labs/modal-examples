# /// script
# dependencies = [
#   "modal==1.1.0",
#   "websockets==15.0.1",
# ]
# ///
# # Stateful Sandbox Code Executor backed by Jupyter
#
# In this example, we demonstrate how to build a Sandbox executor, where the Python
# interpreter state persist. By using a Jupyter kernel, we can distinguish between
# the result of an expression and the output of `stdout` and `stderr`. To run this
# example use: `uv run sandbox_executor.py`.

# ## Defining the `SandboxExecutor`

# First, we define all the dependencies and dataclasses for holding the return
# values from the `SandboxExecutor`.

import json
import re
import secrets
import uuid
from dataclasses import dataclass, field
from http.client import RemoteDisconnected
from time import sleep
from typing import Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import modal
from websockets.sync.client import connect
from websockets.sync.connection import Connection


@dataclass
class ExecutionError:
    name: str
    value: str
    traceback: list[str]


@dataclass
class ExecutionResults:
    result: Optional[str] = None
    logs: list[str] = field(default_factory=list)
    error: Optional[ExecutionError] = None


# The `SandboxExecutor` creates a Modal sandbox and starts a Jupyter Kernel gateway to
# run code in a Jupyter kernel.


@dataclass
class SandboxExecutor:
    """Sandbox executor backed by Modal sandboxes."""

    app_name: str
    packages: list[str] = field(default_factory=list)
    app: modal.App = field(init=False)
    sandbox: modal.Sandbox = field(init=False)
    ws_url: str = field(init=False)

    def __post_init__(self):
        image = modal.Image.debian_slim().uv_pip_install(
            "jupyter_kernel_gateway", "ipykernel", *self.packages
        )
        self.app = modal.App.lookup(self.app_name, create_if_missing=True)
        jupyter_port = 8888
        token = secrets.token_urlsafe(13)
        token_secret = modal.Secret.from_dict({"KG_AUTH_TOKEN": token})

        entrypoint = [
            "jupyter",
            "kernelgateway",
            "--KernelGatewayApp.ip='0.0.0.0'",
            f"--KernelGatewayApp.port={jupyter_port}",
            "--KernelGatewayApp.allow_origin='*'",
        ]
        with modal.enable_output():
            self.sandbox = modal.Sandbox.create(
                *entrypoint,
                app=self.app,
                image=image,
                secrets=[token_secret],
                encrypted_ports=[jupyter_port],
                timeout=60 * 60,
            )

        tunnel = self.sandbox.tunnels()[jupyter_port]
        self._wait_for_server(tunnel.host, token)

        kernel_id = self._start_kernel(tunnel.host, token)
        self.ws_url = (
            f"wss://{tunnel.host}/api/kernels/{kernel_id}/channels?token={token}"
        )

    def run_code(self, code: str) -> ExecutionResults:
        with connect(self.ws_url) as ws:
            return self._run_code(ws, code)

    def terminate(self):
        self.sandbox.terminate()

    @classmethod
    def _send_execute_request(cls, ws: Connection, code: str) -> str:
        msg_id = str(uuid.uuid4())
        execute_request = {
            "header": {
                "msg_id": msg_id,
                "username": "anonymous",
                "session": str(uuid.uuid4()),
                "msg_type": "execute_request",
                "version": "5.0",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
            },
        }
        ws.send(json.dumps(execute_request))
        return msg_id

    @classmethod
    def _run_code(cls, ws: Connection, code: str) -> ExecutionResults:
        """Run code on the websocket connection."""
        msg_id = cls._send_execute_request(ws, code)

        result = None
        logs = []

        while True:
            msg = json.loads(ws.recv())
            parent_msg_id = msg.get("parent_header", {}).get("msg_id")
            if parent_msg_id != msg_id:
                continue
            msg_type = msg.get("msg_type", "")
            msg_content = msg.get("content", {})
            if msg_type == "stream":
                logs.append(msg_content["text"])
            elif msg_type == "execute_result":
                # Only support text for this simple example
                result = msg_content["data"].get("text/plain", None)
            elif msg_type == "error":
                traceback = [
                    _strip_ansi_colors(line)
                    for line in msg_content.get("traceback", [])
                ]
                error = ExecutionError(
                    name=msg_content.get("ename", ""),
                    value=msg_content.get("evalue", ""),
                    traceback=traceback,
                )
                return ExecutionResults(error=error)
            elif msg_type == "status" and msg_content["execution_state"] == "idle":
                break

        return ExecutionResults(result=result, logs=logs)

    @classmethod
    def _wait_for_server(cls, host: str, token: str):
        """Wait for server to start up."""
        counter = 0
        req = Request(f"https://{host}/api/kernelspecs?token={token}", method="GET")
        while True:
            try:
                with urlopen(req):
                    pass
                break
            except (HTTPError, RemoteDisconnected):
                counter += 1
                if counter > 100:
                    raise RuntimeError("Unable to connect to sandbox")
            sleep(1.0)

    @classmethod
    def _start_kernel(cls, host: str, token: str) -> str:
        """Start kernel."""
        kernels_url = f"https://{host}/api/kernels?token={token}"
        req = Request(kernels_url, method="POST")
        with urlopen(req) as response:
            body = response.read()
            json_resp = json.loads(body)

        return json_resp["id"]


ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _strip_ansi_colors(text: str) -> str:
    """Remove ansi colors from text."""
    return ANSI_ESCAPE.sub("", text)


# ## Using the API

# We create a `SandboxExecutor` with an `app_name` and run code using the `run_code`
# API.

executor = SandboxExecutor(app_name="jupyter-executor")

# When running `a = 10` there are no outputs because the code does not have a result,
# error, or evaluating an expression.

output = executor.run_code("a = 10")
print(output)

# When running `a`, the output consist of a `results` that is the value of `a`.

output = executor.run_code("a")
print(output.result)

# When running `print(a + 100)`, the output are logs, because the express prints to
# `stdout`.

output = executor.run_code("print(a + 100)")
print(output.logs)

# When running `a / 0`, the output is an error with the error name, value and traceback.

output = executor.run_code("a/0")
print(output.error)

executor.terminate()
