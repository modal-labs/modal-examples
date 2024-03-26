# ---
# cmd: ["modal", "serve", "10_integrations/autogen/autogen_studio_ui.py"]
# ---
#
# # Run AutoGen Studio UI
#
# This example shows you how to deploy an AutoGen Studio UI app on Modal.
#
import os
import subprocess

import modal
from modal import Image, Secret, Stub

SERVER_DEPLOYMENT_NAME = "autogen-studio-webui"
AUTOGEN_STUDIO_PORT = 8081

MINUTES = 60  # in units of seconds
HOURS = 60 * MINUTES

openai_secret = Secret.from_name(
    "openai-secret"
)  # create a modal.Secret with your OpenAI key, https://modal.com/docs/guide/secrets


# Define the image
webui_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("--no-install-recommends", "build-essential", "npm", "git-lfs")
    .run_commands("npm install -g yarn")
    .pip_install(
        "pydoc-markdown~=4.8.2",
        "autogenstudio==0.0.56",
        "pydantic==2.6.4",
        "fastapi==0.109.2",
        "typer==0.6.1",
        "uvicorn==0.29.0",
        "arxiv~=2.1.0",
        "pyautogen~=0.2.0",
        "websockets==12.0",
        "numpy~=1.25.0",
    )
)

stub = Stub(
    name=SERVER_DEPLOYMENT_NAME,
    image=webui_image,
)


@stub.function(
    secrets=[openai_secret],
)
@modal.web_server(AUTOGEN_STUDIO_PORT)
def run(timeout: int = 8 * HOURS):
    subprocess.Popen(
        "autogenstudio ui"
        + " --host"
        + " 0.0.0.0"
        + " --port"
        + f" {AUTOGEN_STUDIO_PORT}",
        env={
            "OPENAI_API_KEY": str(os.environ.get("OPENAI_API_KEY", None)),
        },
        shell=True,
    )


# Doing `modal run autogen_studio_ui.py` will run a Modal app which starts
# the AutoGen Studio UI server at an address like https://u33iiiyq33klbs.r3.modal.host.
# Visit this address in your browser, and you should see the AutoGen Studio UI.
