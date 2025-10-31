# ---
# cmd: ["modal", "serve", "01_getting_started/inference_endpoint.py"]
# ---
from pathlib import Path

import modal

app = modal.App("example-inference-endpoint")
image = (
    modal.Image.debian_slim()
    .uv_pip_install("transformers[torch]")
    .uv_pip_install("fastapi")
)


@app.function(gpu="h100", image=image)
@modal.fastapi_endpoint(docs=True)
def chat(prompt: str | None = None) -> list[dict]:
    from transformers import pipeline

    if prompt is None:
        prompt = f"/no_think Read this code.\n\n{Path(__file__).read_text()}\nIn one paragraph, what does the code do?"

    print(prompt)
    context = [{"role": "user", "content": prompt}]

    chatbot = pipeline(
        model="Qwen/Qwen3-1.7B-FP8", device_map="cuda", max_new_tokens=1024
    )
    result = chatbot(context)
    print(result[0]["generated_text"][-1]["content"])

    return result
