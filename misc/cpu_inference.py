from modal import App, Image

app = App("cpu-inference")
BATCH_SIZE = 64
NUM_CORES = 64
NUM_OUTPUT_TOKENS = 128

PROMPT = """You are an expert at adding tags to pieces of text. Add a list of comma separated tags to the following pieces of text. Here are some examples:

Example 1

Text: 	
IIJA Bureau of Land Management Idaho Threatened and Endangered Species Program Department of the Interior - Bureau of Land Management Idaho Threatened and Endangered Species Program
Tags: ["Wildlife Conservation", "Environmental Protection", "Species Preservation", "Conservation Efforts", "Ecosystem Management" ]

-------------------

Example 2

Text: Scaling Apprenticeship Readiness Across the Building Trades Initiative A Cooperative Agreement will be awarded for $19,821,832 to TradesFutures to substantially increase the number of participants from underrepresented populations and underserved communities in registered apprenticeship programs within the construction industry sector.	
Tags: [ "Apprenticeship", "Building Trades", "Construction Industry", "Underrepresented Populations", "Underserved Communities" ]

"""

llama_cpp_image = Image.debian_slim(python_version="3.11").apt_install(["curl", "unzip"]).run_commands([
    'curl -L -O https://github.com/ggerganov/llama.cpp/releases/download/b3367/llama-b3367-bin-ubuntu-x64.zip',
    'unzip llama-b3367-bin-ubuntu-x64.zip',
    'curl -L -O https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf',
])

def batch_iterator(dataset):
    for i in range(0, len(dataset), BATCH_SIZE):
        yield dataset[i : i + BATCH_SIZE]["text"]

@app.function(image = llama_cpp_image)
def llama_cpp_inference(batch):
    import subprocess
    import time

    start = time.monotonic()
    # TODO: Add support for batching, check if it's tagging correctly
    subprocess.run([
        '/build/bin/llama-cli', 
        '-m', '/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf', 
        '-b', f'{BATCH_SIZE}', 
        '-n', f'{NUM_OUTPUT_TOKENS}',
        '-p', f'{PROMPT} \n batch'
    ])

    end = time.monotonic()
    return end - start


@app.function(image = Image.debian_slim().pip_install("datasets"))
def process_data():
    from datasets import load_dataset
    dataset = load_dataset("youngermax/text-tagging", split="train")
    max_duration = 0
    for duration in llama_cpp_inference.map(batch_iterator(dataset)):
        max_duration = max(max_duration, duration)

    # TODO: Fix throughput measurement
    print(f"The throughput is f{NUM_OUTPUT_TOKENS * len(dataset) / max_duration}")

@app.local_entrypoint()
def main():
    process_data.remote()
