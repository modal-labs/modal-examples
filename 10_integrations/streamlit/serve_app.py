import asyncio
import os
import pathlib

import modal

image = modal.Image.debian_slim().pip_install("streamlit", "numpy", "pandas")
stub = modal.Stub(name="example-modal-streamlit", image=image)

@stub.function(
    mounts=[
        modal.Mount.from_local_file(pathlib.Path(__file__).parent / "app.py", remote_path="/root/app.py")
    ],
)
def f():
    from streamlit.web.bootstrap import _on_server_start, _set_up_signal_handler, load_config_options
    from streamlit.web.server import Server

    # Create the server. It won't start running yet.
    server = Server("/root/app.py", None)

    async def run_server() -> None:
        # Start the server
        await server.start()
        _on_server_start(server)

        # Install a signal handler that will shut down the server
        # and close all our threads
        _set_up_signal_handler(server)

        # Wait until `Server.stop` is called, either by our signal handler, or
        # by a debug websocket session.
        await server.stopped

    # Run the server. This function will not return until the server is shut down.
    with modal.forward(8501) as tunnel:
        load_config_options({"browser.serverAddress": tunnel.host, "browser.serverPort": 443})
        asyncio.run(run_server())
