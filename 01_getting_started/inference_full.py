# ---
# cmd: ["python", "01_getting_started/inference_full.py"]
# deploy: true
# mypy: ignore-errors
# ---
from pathlib import Path

import modal

app = modal.App("example-inference-full")
image = (
    modal.Image.debian_slim()
    .uv_pip_install("transformers[torch]")
    .uv_pip_install("fastapi")
)

with image.imports():
    from transformers import pipeline

weights_cache = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "example-inference", create_if_missing=True
    )
}


@app.cls(gpu="h100", image=image, volumes=weights_cache, enable_memory_snapshot=True)
class Chat:
    @modal.enter()
    def init(self):
        self.chatbot = pipeline(
            model="Qwen/Qwen3-1.7B-FP8", device_map="cuda", max_new_tokens=1024
        )

    @modal.fastapi_endpoint(docs=True)
    def web(self, prompt: str | None = None) -> list[dict]:
        result = self.run.local(prompt)
        return result

    @modal.method()
    def run(self, prompt: str | None = None) -> list[dict]:
        if prompt is None:
            prompt = f"/no_think Read this code.\n\n{Path(__file__).read_text()}\nIn one paragraph, what does the code do?"

        print(prompt)
        context = [{"role": "user", "content": prompt}]

        result = self.chatbot(context)
        print(result[0]["generated_text"][-1]["content"])

        return result


@app.local_entrypoint()
def main():
    import glob

    chat = Chat()
    root_dir, examples = Path(__file__).parent.parent, []
    for path in glob.glob("**/*.py", root_dir=root_dir):
        examples.append(
            f"/no_think Read this code.\n\n{(root_dir / path).read_text()}\nIn one paragraph, what does the code do?"
        )

    for result in chat.run.map(examples):
        print(result[0]["generated_text"][-1]["content"])


if __name__ == "__main__":
    import json
    import urllib.request
    from datetime import datetime

    ChatCls = modal.Cls.from_name(app.name, "Chat")
    chat = ChatCls()
    print(datetime.now(), "making .remote call to Chat.run")
    print(chat.run.remote())
    print(datetime.now(), "making web request to", url := chat.web.get_web_url())

    with urllib.request.urlopen(url) as response:
        print(datetime.now())
        print(json.loads(response.read().decode("utf-8")))
