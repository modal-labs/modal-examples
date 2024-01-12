import modal

# First run `modal volume create my-hf-cache` in CLI first.
model_vol = modal.Volume.lookup("my-hf-cache")
hf_secret = modal.Secret.lookup("huggingface")

Model = modal.Cls.lookup(
    "example-vllm-generic", "Model", workspace="modal-labs"
)

Model_40GB = Model.with_options(
    secrets=[hf_secret],
    gpu=modal.gpu.A100(memory=40),
    volumes={"/hf-cache": model_vol},
    allow_background_volume_commits=True,
)

Model_80GB = Model.with_options(
    secrets=[hf_secret],
    gpu=modal.gpu.A100(memory=80),
    volumes={"/hf-cache": model_vol},
    allow_background_volume_commits=True,
)

mistral7b = Model_40GB(model_name="mistralai/Mistral-7B-Instruct-v0.2")
llama7b = Model_80GB(model_name="meta-llama/Llama-2-7b-chat-hf")

prompt = "[INST] Can you code Dijkstra in Rust? [/INST] "

print(f"Sending prompt to LLaMA 7B\n*** {prompt} ***")
for text in llama7b.completion_stream.remote_gen(prompt):
    print(text, end="", flush=True)

print(f"Sending prompt to Mistral 7B\n*** {prompt} ***")
for text in mistral7b.completion_stream.remote_gen(prompt):
    print(text, end="", flush=True)
