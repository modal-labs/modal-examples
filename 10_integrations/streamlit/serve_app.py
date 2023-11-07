import pathlib

import modal

image = modal.Image.debian_slim().pip_install("streamlit", "numpy", "pandas")
stub = modal.Stub(name="example-modal-streamlit", image=image)
stub.q = modal.Queue.new()

session_timeout = 15 * 60

@stub.function(
    mounts=[
        modal.Mount.from_local_file(pathlib.Path(__file__).parent / "app.py", remote_path="/root/app.py")
    ],
)
def run_streamlit(publish_url: bool = False):
    from streamlit.web.bootstrap import load_config_options, run
    # Run the server. This function will not return until the server is shut down.
    with modal.forward(8501) as tunnel:
        # Reload Streamlit config with information about Modal tunnel address.
        if publish_url:
            stub.q.put(tunnel.url)
        load_config_options({"browser.serverAddress": tunnel.host, "browser.serverPort": 443})
        run(
            main_script_path="/root/app.py",
            command_line=None,
            args=["--timeout", str(session_timeout)],
            flag_options={},
        )


@stub.function()
@modal.web_endpoint(method="GET")
def spawn():
    from fastapi.responses import RedirectResponse
    run_streamlit.spawn(publish_url=True)
    url = stub.q.get()
    return RedirectResponse(url, status_code=303)
