# ---
# lambda-test: false
# ---
#
# # Run and share Streamlit apps
#
# This example supports running the Streamlit app ephemerally with `modal run`, and
# deploying a web endpoint from which others can spin up isolated instances of the Streamlit
# app, each accessible in the browser via URL!
#
# ![example streamlit app](./streamlit.png)
#
# The example is structured as two files:
#
# 1. This module, which defines the Modal objects (name the script `serve_streamlit.py` locally).
# 2. `app.py`, which is a Streamlit script and is mounted into a Modal function ([download script](https://github.com/modal-labs/modal-examples/blob/main/10_integrations/streamlit/app.py)).
import pathlib

import modal

# ## Define container dependencies
#
# The `app.py` script imports three third-party packages, so we include these in the example's
# image definition.

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("streamlit", "numpy", "pandas")
    # Use fork until https://github.com/valohai/asgiproxy/pull/11 is merged.
    .pip_install("git+https://github.com/modal-labs/asgiproxy.git")
)

stub = modal.Stub(name="example-modal-streamlit", image=image)

streamlit_script_local_path = pathlib.Path(__file__).parent / "app.py"
streamlit_script_remote_path = pathlib.Path("/root/app.py")

# ## Mounting the `app.py` script
#
# As the Modal code and Streamlit code are isolated, we can just mount the latter into the container
# at a configured path, and pass that path to the Streamlit server.
#
# We could also import the module, and then pass `app.__path__` to Streamlit.

HOST = "127.0.0.1"
PORT = "8000"


def spawn_server():
    import socket
    import subprocess

    process = subprocess.Popen(
        [
            "streamlit",
            "run",
            str(streamlit_script_remote_path),
            "--browser.serverAddress",
            HOST,
            "--server.port",
            PORT,
            "--browser.serverPort",
            PORT,
            "--server.enableCORS",
            "false",
        ]
    )

    # Poll until webserver accepts connections before running inputs.
    while True:
        try:
            socket.create_connection((HOST, int(PORT)), timeout=1).close()
            print("Webserver ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webserving process has exited.
            # If so, a connection can never be made.
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(
                    f"launcher exited unexpectedly with code {retcode}"
                )


streamlit_script_mount = modal.Mount.from_local_file(
    streamlit_script_local_path,
    streamlit_script_remote_path,
)


@stub.function(
    allow_concurrent_inputs=100,
    mounts=[streamlit_script_mount],
)
@modal.asgi_app()
def run():
    from asgiproxy.config import BaseURLProxyConfigMixin, ProxyConfig
    from asgiproxy.context import ProxyContext
    from asgiproxy.simple_proxy import make_simple_proxy_app

    spawn_server()

    config = type(
        "Config",
        (BaseURLProxyConfigMixin, ProxyConfig),
        {
            "upstream_base_url": f"http://{HOST}:{PORT}",
            "rewrite_host_header": f"{HOST}:{PORT}",
        },
    )()
    proxy_context = ProxyContext(config)
    return make_simple_proxy_app(proxy_context)
