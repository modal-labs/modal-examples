# ---
# cmd: ["modal", "run", "-m", "13_sandboxes.test_case_generator::main"]
# ---
import modal

app = modal.App(name="sandbox-test-case-generator")
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
GH_OWNER = "modal-labs"
GH_REPO_NAME = "password-analyzer"
GH_MODULE_NAME = "password_strength"
GH_BRANCH = "main"


@app.cls(
    image=model_image,
    volumes={
        "/cache": model_volume,
        "/data": files_volume,
    },
    gpu="L40S",
)
class Deepseek:
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
        print(f"Output written to {output_file_name}")
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
        print("Applying chat template...")
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        print("Generating...")
        outputs = self.model.generate(
            inputs,
            max_new_tokens=1024,
            do_sample=False,
            num_return_sequences=1,
        )
        print("Decoding...")
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
def download_files_to_volume(folder_paths: list[str]) -> list[str]:
    import os

    import requests

    os.makedirs("/data/inputs", exist_ok=True)
    all_files = []
    for folder_path in folder_paths:
        response = requests.get(
            f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO_NAME}/contents/{folder_path}?ref={GH_BRANCH}"
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

    print("Files downloaded to volume.")
    return [name for name in file_to_text.keys() if not name.startswith("test_")]


sb_image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl")
    .pip_install("modal")
    .run_commands(
        "curl -sSL https://install.python-poetry.org | python3 -",
    )
    .env({"PATH": "$PATH:/root/.local/bin"})
)


def run_sandbox(file_name: str):
    sb = modal.Sandbox.create(
        app=app,
        image=sb_image,
        volumes={"/data": files_volume},
    )
    print(f"Sandbox created with object ID: {sb.object_id}")

    p = sb.exec("git", "clone", f"https://github.com/{GH_OWNER}/{GH_REPO_NAME}")
    print(p.stdout.read())
    print(p.stderr.read())

    # Verify that tests initially pass
    p = sb.exec(
        "bash",
        "-c",
        "cd password-analyzer && poetry install --no-root && poetry run pytest",
    )
    print(p.stdout.read())
    print(p.stderr.read())

    # Run the generated test file
    p = sb.exec(
        "bash",
        "-c",
        f"cp /data/outputs/{file_name} password-analyzer/tests/ && cd password-analyzer && poetry run pytest",
    )
    print(p.stdout.read())
    print(p.stderr.read())

    sb.terminate()


@app.function(volumes={"/data": files_volume})
def main():
    deepseek = Deepseek()
    input_files = download_files_to_volume.remote(
        folder_paths=["src/password_strength", "tests"]
    )

    print("Generating tests for: ", input_files)
    output_files = deepseek.generate.map(input_files)
    for output_file in output_files:
        print(output_file)
        run_sandbox(output_file)


# # Addenda
# The below functions are utility functions.


def get_user_prompt(file_text: str, test_file_text: str) -> str:
    return f"""
    You are an expert Python test engineer. Your task is to improve an existing test file using `pytest`.

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
