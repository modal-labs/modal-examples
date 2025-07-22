# ---
# cmd: ["modal", "run", "-m", "13_sandboxes.test_case_generator"]
# args: ["--gh-owner", "modal-labs", "--gh-repo-name", "password-analyzer", "--gh-module-path", "src/password_strength", "--gh-test-dir-path", "tests", "--gh-branch", "main"]
# ---
import modal

app = modal.App(
    name="sandbox-test-case-generator",
)
model_volume = modal.Volume.from_name("deepseek-model-volume", create_if_missing=True)
files_volume = modal.Volume.from_name("files-volume", create_if_missing=True)


model_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "transformers==4.53.2",
        "torch==2.7.1",
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

with model_image.imports() as imports:
    from transformers import AutoModelForCausalLM, AutoTokenizer


# GH Repo Configs
# GH_OWNER = "modal-labs"
# GH_REPO_NAME = "password-analyzer"
# GH_MODULE_NAME = "password_strength"
# GH_BRANCH = "main"


@app.cls(
    image=model_image,
    volumes={
        "/cache": model_volume,
        "/data": files_volume,
    },
    gpu="L40S",
)
class TestFileGenerator:
    @modal.enter()
    def load_model(self):
        MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
        REVISION = "e5d64addd26a6a1db0f9b863abf6ee3141936807"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=REVISION)
        self.model.to("cuda")

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
        file_contents, test_file_contents = self.load_inputs(file_name)

        system_prompt = get_system_prompt()
        user_prompt = get_user_prompt(file_contents, test_file_contents)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        outputs = self.model.generate(
            inputs,
            max_new_tokens=1024,
            do_sample=False,
            num_return_sequences=1,
        )
        model_output = self.tokenizer.decode(
            outputs[0][len(inputs[0]) :], skip_special_tokens=True
        )
        output_contents = post_process(model_output)
        return self.write_outputs(f"test_{file_name}", output_contents)


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install(
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
        "mkdir -p allure-results &&"
        + f"webdiff password-analyzer/tests/{file_name} /data/outputs/{file_name}  --host 0.0.0.0 --port 8001 &&"
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
    )
    return sb


@app.local_entrypoint()
def main(
    gh_owner: str,  # = "modal-labs",
    gh_repo_name: str,  # = "password-analyzer",
    gh_module_path: str,  # = "src/password_strength",
    gh_test_dir_path: str,  # = "tests",
    gh_branch: str,  # = "main",
):
    deepseek = TestFileGenerator()
    input_files = download_files_to_volume.remote(
        folder_paths=[gh_module_path, gh_test_dir_path],
        gh_owner=gh_owner,
        gh_repo_name=gh_repo_name,
        gh_branch=gh_branch,
    )
    output_files = list(deepseek.generate.map(input_files))
    sandboxes = create_sandboxes(output_files, gh_owner, gh_repo_name)
    poll_sandboxes(sandboxes)


# # Addenda
# The below functions are utility functions.
def create_sandboxes(filenames: list[str], gh_owner: str, gh_repo_name: str):
    import time

    file_to_sandbox: dict[str, modal.Sandbox] = {}
    for filename in filenames:
        print(f"Running sandbox for {filename}")
        image = get_sandbox_image(gh_owner, gh_repo_name)
        sb = run_sandbox(image, filename)
        file_to_sandbox[filename] = sb
    time.sleep(20)

    for filename, sb in file_to_sandbox.items():
        tunnel1 = sb.tunnels()[8000]
        tunnel2 = sb.tunnels()[8001]
        print(f"Sandbox created and run for generated test file: {filename}")
        print(f"✨ View diff: {tunnel2.url}")
        print(f"✨ View test results: {tunnel1.url}\n")

    return file_to_sandbox.values()


def poll_sandboxes(sandboxes: list[modal.Sandbox]):
    """
    Poll sandboxes every 10 seconds until all are completed.
    """
    import time

    completed_sandbox_ids = set()
    while len(completed_sandbox_ids) < len(sandboxes):
        for sb in sandboxes:
            if sb.poll() is not None:
                print(f"Sandbox {sb.object_id} completed")
                completed_sandbox_ids.add(sb.object_id)
        time.sleep(10)


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
    - This file will be run directly using `pytest`, so it must be immediately runnable.

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
        "You write clean, idiomatic unit tests using the `pytest` framework, based on source code with functions or classes. "
        "Your task is to enhance an existing test file by adding more test cases. Focus on edge cases, input validation, and untested behavior, especially as inferred from docstrings and type hints. "
        "Do not change or add import statements. Do not explain your reasoning. Output only a complete, valid Python file. "
        "Limit each line to a maximum of 100 characters to avoid output truncation or formatting errors."
    )


def post_process(output: str) -> str:
    """
    Remove LLM code block formatting (e.g., ```python ... ```).
    Specifically:
    - Removes everything before and including the first ```python
    - Removes everything after and including the last ```
    - If neither tag exists, return the output unchanged
    """
    lines = output.splitlines()

    # Locate ```python and ``` tags
    start_idx = next(
        (i + 1 for i, line in enumerate(lines) if line.strip().startswith("```python")),
        None,
    )
    end_idx = next(
        (i for i in reversed(range(len(lines))) if lines[i].strip() == "```"), None
    )

    # If either tag is missing, return original output
    if start_idx is None or end_idx is None or start_idx >= end_idx:
        return output

    return "\n".join(lines[start_idx:end_idx])
