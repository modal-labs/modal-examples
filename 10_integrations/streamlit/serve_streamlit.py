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
# 2. `app.py`, which is a Streamlit script and is mounted into a Modal function ([download script](./app.py)).
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
)
stub = modal.Stub(name="example-modal-streamlit", image=image)
stub.q = modal.Queue.new()

# ## Sessions
#
# The Streamlit app is executed within Modal using the `run_streamlit` Modal function.
# Every Modal function has a configurable timeout, and we set this function's timeout to
# 15 minutes. If your Streamlit app users want to spend longer in a single session, you can
# extend the function's timeout [up to 24 hours](/docs/guide/timeouts#timeouts).

session_timeout = 15 * 60
streamlit_script_local_path = pathlib.Path(__file__).parent / "app.py"
streamlit_script_remote_path = pathlib.Path("/root/app.py")

# ## Mounting the `app.py` script
#
# As the Modal code and Streamlit code are isolated, we can just mount the latter into the container
# at a configured path, and pass that path to the Streamlit server.
#
# We could also import the module, and then pass `app.__path__` to Streamlit.


@stub.function(
    mounts=[
        modal.Mount.from_local_file(
            streamlit_script_local_path,
            remote_path=streamlit_script_remote_path,
        )
    ],
    timeout=session_timeout,
)
def run_streamlit(publish_url: bool = False):
    from streamlit.web.bootstrap import load_config_options, run

    # Run the server. This function will not return until the server is shut down.
    with modal.forward(8501) as tunnel:
        # Reload Streamlit config with information about Modal tunnel address.
        if publish_url:
            stub.q.put(tunnel.url)
        load_config_options(
            {"browser.serverAddress": tunnel.host, "browser.serverPort": 443}
        )
        run(
            main_script_path=str(streamlit_script_remote_path),
            command_line=None,
            args=["--timeout", str(session_timeout)],
            flag_options={},
        )


# ## Sharing
#
# Deploy this Modal app and you get a web endpoint URL that users can hit in their browser
# to get their very own instance of the Streamlit app.
#
# The shareable URL for this app is https://modal-labs--example-modal-streamlit-share.modal.run.
#
# This technique is very similar to what is shown in the [Tunnels guide](/docs/guide/tunnels).


@stub.function()
@modal.web_endpoint(method="GET")
def share():
    from fastapi.responses import RedirectResponse

    run_streamlit.spawn(publish_url=True)
    url = stub.q.get()
    return RedirectResponse(url, status_code=303)
