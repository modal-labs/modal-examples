from modal import Stub, Image, method

# vLLM uses default HF cache, so we can use HF to pre-download the model into our image.
# This snippet is optional but avoids re-downloading from Huggingface by using Modal's CDN (faster cold starts!).
def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download("lmsys/vicuna-13b-v1.3")
    snapshot_download("hf-internal-testing/llama-tokenizer")

image = (
    Image
    .from_dockerhub("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install("torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118")
    .pip_install("vllm @ git+https://github.com/vllm-project/vllm.git@2b7d3aca2e1dd25fe26424f57c051af3b823cd71")
    .run_function(download_model)
)

stub = Stub(image=image)

@stub.cls(gpu="A100")
class Model:
    def __enter__(self):
        from vllm import LLM

        self.llm = LLM(model="lmsys/vicuna-13b-v1.3") # Load the model
    
    @method()
    def generate(self):
        from vllm import SamplingParams

        prompts = ["USER:\nImplement a Python function to compute the Fibonacci numbers.\n\nASSISTANT:\n"] # You can put several prompts in this list
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
        result = self.llm.generate(prompts, sampling_params) # Trigger inference
        print(result[0].outputs[0].text)

@stub.local_entrypoint()
def main():
    model = Model()
    model.generate.call()
