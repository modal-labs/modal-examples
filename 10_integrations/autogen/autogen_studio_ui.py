# ---
# lambda-test: false
# ---
#
# # Run AutoGen Studio UI
#
# This example shows you how to run an AutoGen Studio UI app with `modal serve`, and then deploy it as a serverless web app.
#
# <img src="./autogen-logo.svg" style="width:2rem;height:2rem">
#
import os
import subprocess
import time
import modal

from pathlib import Path

from modal import Stub, Image, Secret, Mount

# from dotenv import load_dotenv # Optional

server_deployment_name = "autogen-studio-webui"
autogen_studio_port = 8081

# Optionally mount a .env file
env_local_path = Path(__file__).parent / ".env"
env_remote_path = Path("/root/.env")
env_mount = Mount.from_local_file(
    local_path=env_local_path,
    remote_path=env_remote_path.as_posix(),
)

# Optionally use Modal Secrets if you have a secret dictionary setup
secret = Secret.from_name("LLM-Services")


# Define the dependencies for the image
dependencies = [
    "pydoc-markdown",
    "autogenstudio",
    "pydantic>=2,<3,!=2.6.0",
    # "pydantic_settings>=2.0.0", # Optional
    "fastapi==0.109.2",
    "typer",
    "uvicorn",
    "arxiv",
    "pyautogen>=0.2.0",
    "python-dotenv",
    "websockets",
    "numpy < 2.0.0",
]

# Define the image
webui_image = (
    Image.debian_slim("3.10")
        .env({"DEBIAN_FRONTEND": "noninteractive"})
        .run_commands("curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash")
        .apt_install("--no-install-recommends", "build-essential", "npm", "git-lfs")
        .run_commands("apt-get autoremove", "apt-get clean")
        .run_commands("rm -rf /var/lib/apt/lists/*")
        .env({"DEBIAN_FRONTEND": "dialog"})
        .run_commands("npm install -g yarn")
        .pip_install(dependencies)
)

stub = Stub(
    name=server_deployment_name,
    image=webui_image,
)

@stub.function(
    concurrency_limit=3,
    timeout=1_500,
    mounts=[env_mount],
    secrets=[secret],
)
@modal.web_server(autogen_studio_port)
def run_autogen_studio(timeout: int = 10_000):
    # load_dotenv(env_remote_path) # Optional
    cmd = f"autogenstudio ui --host 0.0.0.0 --port {autogen_studio_port}"
    with modal.forward(autogen_studio_port) as autogen_studio_tunnel:
        print(f"Autogen Studio is running at: {autogen_studio_tunnel.url}")
        autogen_process = subprocess.Popen(
            cmd,
            env={
                # Optional
                # **os.environ,
                # "OPENAI_API_KEY": str(os.environ.get("OPENAI_API_KEY", None)),
            },
            shell=True,
        )
        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
            print(f"Reached end of {timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            autogen_process.kill()


@stub.local_entrypoint()
def main(timeout: int = 10_000):
    # Run Autogen Studio
    run_autogen_studio.remote(timeout=timeout)


# Doing `modal run autogen_studio_ui.py` will run a Modal app which starts
# the AutoGen Studio UI server at an address like https://u33iiiyq33klbs.r3.modal.host.
# Visit this address in your browser, and you should see the AutoGen Studio UI.


