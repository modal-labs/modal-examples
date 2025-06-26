import modal

image = (
    modal.Image.debian_slim()
    .pip_install(
        "smolagents==1.18.0",
        "vllm",
        "torch",
        "huggingface-hub[hf_transfer]==0.31.2",
        "hf_xet==1.1.1",
        "sqlalchemy",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("modal_executor")
)
app = modal.App("smolagent-text-to-sql")

hf_cache_vol = modal.Volume.from_name("smol-agent-huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("smol-agent-vllm-cache", create_if_missing=True)


@app.cls(
    cpu=4,
    memory=8192,
    gpu="A10G",
    timeout=10 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    image=image,
)
class MyAgent:
    @modal.enter()
    def create_agent(self):
        from smolagents import VLLMModel, CodeAgent
        from modal_executor import get_executor_cls

        model = VLLMModel(model_id="Qwen/Qwen2.5-Coder-7B-Instruct")
        ModelExecutor = get_executor_cls()

        # If we upstream a modal exeuctor, then we can write:
        # agent = CodeAgent(tools=[], model=model, executor_type="modal")
        agent = CodeAgent(tools=[], model=model)
        agent.python_executor = ModelExecutor([], agent.logger, app=app)

        self.agent = agent

    @modal.method()
    def run_agent(self):
        result = self.agent.run("Can you give me the 100th Fibonacci number?")
        print(result)
        return result
