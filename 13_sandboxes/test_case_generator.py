# ---
# cmd: ["modal", "run", "-m", "13_sandboxes.test_case_generator"]
# args: ["--gh-owner", "modal-labs", "--gh-repo-name", "password-analyzer", "--gh-module-path", "src/password_strength", "--gh-tests-path", "tests", "--gh-branch", "main"]
# ---
import subprocess
import time

import modal

app = modal.App(
    name="sandbox-test-case-generator",
)
model_volume = modal.Volume.from_name("deepseek-model-volume", create_if_missing=True)
files_volume = modal.Volume.from_name("files-volume", create_if_missing=True)

MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
MODEL_REVISION = "e5d64addd26a6a1db0f9b863abf6ee3141936807"


model_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.4.9.post3-cu126", add_python="3.12")
    .uv_pip_install(
        "sglang[all]==0.4.9.post3",
        "accelerate==1.8.1",
        "hf_transfer==0.1.9",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache",
        }
    )
    .entrypoint([])  # silence noisy logs
)


@app.cls(
    image=model_image,
    volumes={
        "/cache": model_volume,
        "/data": files_volume,
    },
    gpu="L40S",
    timeout=600,
)
@modal.concurrent(max_inputs=3)  # Each container runs up to 3 requests at once.
class TestCaseServer:
    @modal.enter()
    def download_model(self):
        from huggingface_hub import snapshot_download

        snapshot_download(
            MODEL_NAME,
            local_dir=f"/cache/{MODEL_NAME}",  # similar to cache_dir, but with less unused metadata
            revision=MODEL_REVISION,
            ignore_patterns=["*.pt", "*.bin"],
        )

    @modal.enter()
    def start_model_server(self):
        import subprocess

        serve_params = {
            "host": "0.0.0.0",
            "port": 8000,
            "model": f"/cache/{MODEL_NAME}",
            "log-level": "error",
        }
        serve_cmd = "python -m sglang.launch_server " + " ".join(
            [f"--{k} {v}" for k, v in serve_params.items()]
        )

        self.serve_process = subprocess.Popen(serve_cmd, shell=True)
        wait_for_port(self.serve_process, 8000)

        print("SGLang server is ready!")

    @modal.web_server(port=8000, startup_timeout=240)
    def serve(self):
        return


@app.cls(
    image=modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "openai==1.97.1"
    ),
    volumes={
        "/data": files_volume,
    },
)
class TestCaseClient:
    url: str = modal.parameter()

    def load_inputs(self, file_name: str) -> tuple[str, str]:
        import os

        if not os.path.exists("/data/inputs"):
            raise Exception(
                "Inputs directory does not exist. Make sure to run download_files_to_volume first."
            )

        with open(f"/data/inputs/{file_name}", "r") as f:
            file_contents = f.read()

        with open(f"/data/inputs/test_{file_name}", "r") as f:
            test_file_contents = f.read()
        return file_contents, test_file_contents

    def write_outputs(self, output_file_name: str, output_contents: str) -> str:
        import os

        os.makedirs("/data/outputs", exist_ok=True)
        with open(f"/data/outputs/{output_file_name}", "w") as f:
            f.write(output_contents)
        return output_file_name

    @modal.method()
    def generate(self, file_name: str) -> str:
        import json

        import openai

        file_contents, test_file_contents = self.load_inputs(file_name)

        system_prompt = get_system_prompt()
        user_prompt = get_user_prompt(file_contents, test_file_contents)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        client = openai.Client(base_url=f"{self.url}/v1", api_key="EMPTY")

        json_schema = {
            "type": "object",
            "properties": {"file_contents": {"type": "string"}},
            "required": ["file_contents"],
        }

        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=0,
            max_tokens=1024,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "test_file",
                    "schema": json_schema,
                },
            },
        )
        output = response.choices[0].message.content
        output_contents = json.loads(output)["file_contents"]
        return self.write_outputs(f"test_{file_name}", output_contents)


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "requests==2.32.3"
    ),
    volumes={"/data": files_volume},
)
def download_files_to_volume(
    folder_paths: list[str],
    gh_owner: str,
    gh_repo_name: str,
    gh_branch: str,
) -> list[str]:
    import os

    import requests

    os.makedirs("/data/inputs", exist_ok=True)
    all_files = []
    for folder_path in folder_paths:
        response = requests.get(
            f"https://api.github.com/repos/{gh_owner}/{gh_repo_name}/contents/{folder_path}?ref={gh_branch}"
        )
        files = response.json()
        all_files.extend(files)

    file_to_download_urls = []
    for _file in all_files:
        if (
            _file["type"] == "file"
            and _file["name"].endswith(".py")
            and _file["name"] != "__init__.py"
        ):
            file_to_download_urls.append((_file["name"], _file["download_url"]))

    file_to_text = {}
    for name, url in file_to_download_urls:
        response = requests.get(url)
        file_to_text[name] = response.text

    for name, text in file_to_text.items():
        with open(f"/data/inputs/{name}", "w") as f:
            f.write(text)
    print("Files downloaded to volume!")
    return [name for name in file_to_text.keys() if not name.startswith("test_")]


def get_sandbox_image(gh_owner: str, gh_repo_name: str):
    ALLURE_VERSION = "2.34.1"
    MODULE_URL = f"https://github.com/{gh_owner}/{gh_repo_name}"

    image = (
        modal.Image.debian_slim()
        .apt_install("git", "curl", "tar", "default-jre")
        .pip_install("webdiff")
        .run_commands(
            f"git clone {MODULE_URL}",
            "curl -sSL https://install.python-poetry.org | python3 -",
            "mkdir -p /opt/allure",
            f"curl -sL https://github.com/allure-framework/allure2/releases/download/{ALLURE_VERSION}/allure-{ALLURE_VERSION}.tgz | tar xz -C /opt/allure --strip-components=1",
        )
        .env({"PATH": "$PATH:/root/.local/bin:/opt/allure/bin"})
    )

    return image


def run_sandbox(image: modal.Image, file_name: str):
    new_file_name = file_name.replace(".py", "_llm.py")

    cmd = (
        f"webdiff password-analyzer/tests/{file_name} /data/outputs/{file_name}  --host 0.0.0.0 --port 8001 &&"
        + "cd password-analyzer && "
        + "poetry install --no-root && "
        + "poetry run pytest --alluredir allure-results || true && "
        + f"cp /data/outputs/{file_name} tests/{new_file_name} && "
        + "poetry run pytest --alluredir allure-results || true && "
        + "allure serve allure-results --host 0.0.0.0 --port 8000"
    )

    sb = modal.Sandbox.create(
        "sh",
        "-c",
        cmd,
        app=app,
        image=image,
        volumes={
            "/data": files_volume,
        },
        encrypted_ports=[8000, 8001],
        timeout=300,  # 5 minutes
    )
    return sb


@app.local_entrypoint()
async def main(
    gh_owner: str,
    gh_repo_name: str,
    gh_module_path: str,
    gh_tests_path: str,
    gh_branch: str,
):
    import asyncio

    # Start server
    sg_lang_server = TestCaseServer()

    # Download files to volume
    input_files = download_files_to_volume.remote(
        folder_paths=[gh_module_path, gh_tests_path],
        gh_owner=gh_owner,
        gh_repo_name=gh_repo_name,
        gh_branch=gh_branch,
    )

    # Initialize client and generate test files
    generator = TestCaseClient(url=sg_lang_server.serve.get_web_url())  # type: ignore
    output_generator = generator.generate.map.aio(input_files)
    output_files = []
    async for f in output_generator:
        if f is not None:
            output_files.append(f)
    print("Test case files generated successfully! Creating sandboxes...")

    # Create sandboxes to run the generated test files
    sandboxes = create_sandboxes(output_files, gh_owner, gh_repo_name)
    await asyncio.gather(
        *[sb.wait.aio(raise_on_termination=False) for sb in sandboxes],
        return_exceptions=True,
    )


# # Addenda
# The below functions are utility functions.
def create_sandboxes(filenames: list[str], gh_owner: str, gh_repo_name: str):
    file_to_sandbox: dict[str, modal.Sandbox] = {}
    for filename in filenames:
        print(f"Running sandbox for {filename}")
        image = get_sandbox_image(gh_owner, gh_repo_name)
        sb = run_sandbox(image, filename)
        file_to_sandbox[filename] = sb
    time.sleep(20)  # Hack to make sure URLs show up at the very end

    for filename, sb in file_to_sandbox.items():
        tunnel1 = sb.tunnels()[8000]
        tunnel2 = sb.tunnels()[8001]
        print(f"Sandbox created and run for generated test file: {filename}")
        print(f"✨ View diff: {tunnel2.url}")
        print(f"✨ View test results: {tunnel1.url}\n")

    return file_to_sandbox.values()


def get_user_prompt(file_text: str, test_file_text: str) -> str:
    return f"""
    Your task is to improve an existing test file using `pytest`.

    Step-by-step:
    1. Carefully read the existing test file (below) and understand the current test cases.
    2. Then read the source file (also below) and understand the function behavior, focusing on docstrings, edge cases, and argument types.
    3. Based on that understanding, **add** new test cases to the test file to increase coverage—especially edge cases, boundary conditions, and untested branches.
    4. Use `pytest` idioms, but do **not** add or change import statements—**use only what is already imported**.
    5. Do **not** explain your reasoning—just return the final modified test file.

    ### Requirements:
    - Your output must be a valid, complete Python file with the added test cases.
    - Do not modify existing test logic unless necessary to support your new test cases.
    - Do not import any additional modules.
    - Limit each line to a maximum of 100 characters to avoid output truncation or formatting errors.
    - Limit your output to around 25 lines. Make sure to complete any functions or blocks you start.


    --- BEGIN TEST FILE ---
    {test_file_text}
    --- END TEST FILE ---

    --- BEGIN SOURCE FILE ---
    {file_text}
    --- END SOURCE FILE ---
    """


def get_system_prompt():
    return (
        "You are a senior software engineer with expertise in test-driven development and Python unit testing. "
        "Your task is to enhance an existing test file by adding more test cases. "
        "Do not change or add import statements. Do not explain your reasoning. Output only a complete, valid Python file. "
        "Do not change existing code and only add new test cases that follow the same formatting as the existing test cases. "
        "Limit each line to a maximum of 100 characters to avoid output truncation or formatting errors."
        "Limit your output to around 25 lines. Make sure to complete any functions or blocks you start."
    )


def wait_for_port(process: subprocess.Popen, port: int):
    import socket

    while True:
        try:
            with socket.create_connection(("0.0.0.0", port), timeout=1):
                break
        except (ConnectionRefusedError, OSError):
            if process.poll() is not None:
                raise Exception(
                    f"Process {process.pid} exited with code {process.returncode}"
                )
