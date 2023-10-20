import os
import socket
import subprocess
import sys
import time

from modal import Dict, Image, Secret, Stub, asgi_app, forward


stub = Stub("example-a1111")
stub.urls = Dict.new()  # TODO: Hack because spawn() doesn't support generators.


def wait_for_port(port: int):
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=5.0):
                break
        except OSError:
            time.sleep(0.1)


@stub.function(
    image=Image.debian_slim()
    .apt_install(
        "wget",
        "git",
        "libgl1",
        "libglib2.0-0",
        "google-perftools",  # For tcmalloc
    )
    .env({"LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"})
    .run_commands(
        "git clone --depth 1 --branch v1.6.0 https://github.com/AUTOMATIC1111/stable-diffusion-webui /webui",
        "python -m venv /webui/venv",
        "cd /webui && . venv/bin/activate && "
        + "python -c 'from modules import launch_utils; launch_utils.prepare_environment()' --xformers",
        gpu="a10g",
    )
    .run_commands(
        "cd /webui && . venv/bin/activate && "
        + "python -c 'from modules import shared_init, initialize; shared_init.initialize(); initialize.initialize()'",
    ),
    gpu="a10g",
    cpu=2,
    memory=1024,
    timeout=3600,
)
def start_web_ui(key: str = "", timeout: int = 10):
    START_COMMAND = r"""
cd /webui && \
. venv/bin/activate && \
accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    --dynamo_backend=inductor \
    --num_cpu_threads_per_process=6 \
    /webui/launch.py \
        --skip-prepare-environment \
        --listen \
        --port 8000
"""
    with forward(8000) as tunnel:
        p = subprocess.Popen(START_COMMAND, shell=True)
        wait_for_port(8000)
        print("[MODAL] ==> Accepting connections at", tunnel.url)
        if key:
            stub.urls[key] = tunnel.url
        p.wait(timeout=timeout * 60.0)


@stub.function(secrets=[Secret.from_name("example-a1111-secret")])
@asgi_app()
def launcher():
    import uuid

    from fastapi import FastAPI, Form, HTTPException
    from fastapi.responses import HTMLResponse, RedirectResponse

    app_password = os.environ["PASSWORD"]
    if not app_password:
        raise ValueError("PASSWORD environment variable must be set in secret")

    app = FastAPI()

    @app.get("/")
    def index():
        stats = start_web_ui.get_current_stats()
        return HTMLResponse(
            rf"""
<!doctype html>
<html>
  <head>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Raleway:400,300,600">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css" />
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css" />
  </head>
  <body class="container">
    <section style="margin-block: 48px">
      <h2 class="title" style="margin-bottom: 16px">Modal A1111 Launcher</h2>
      <p>
        Currently running {stats.num_active_runners} container(s), with {stats.backlog} in the backlog.
        May take up to a minute to start!
      </p>
      <form method="POST" action="/launch">
        <div class="row">
          <label for="i-timeout">Runtime (minutes)</label>
          <input required style="width: 240px" type="number" min="1" max="60" value="10" name="timeout" id="i-timeout">
        </div>
        <div class="row">
          <label for="i-password">Password</label>
          <input required style="width: 240px" type="password" name="password" id="i-password">
        </div>
        <input class="button-primary" type="submit" value="Submit">
      </form>
    </section>
  </body>
</html>
""",
            headers={"Cache-Control": "public, max-age=0, must-revalidate"},
        )

    @app.post("/launch")
    def launch(password: str = Form(), timeout: int = Form()):
        if password != app_password:
            raise HTTPException(401, "Incorrect password")

        key = str(uuid.uuid4())  # for Dict polling
        start_web_ui.spawn(key, timeout)

        while not key in stub.urls:
            time.sleep(0.1)

        return RedirectResponse(stub.urls[key], status_code=303)

    return app
