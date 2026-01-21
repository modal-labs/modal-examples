import modal
from pathlib import Path

# 1. Define the environment (Image)
image = modal.Image.debian_slim().pip_install("transformers", "torch")

app = modal.App("example-inference")

@app.function(
    gpu="any",      # Request a GPU
    image=image,    # Use the custom image defined above
)
def describe_code(file_path: str):
    """
    Uses an AI model to read a local file and summarize it.
    """
    from transformers import pipeline

    # Read the content of the file
    code_content = Path(file_path).read_text()
    
    # Initialize a small, fast model
    model = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # AI processes the code
    summary = model(code_content, max_length=130, min_length=30, do_sample=False)
    
    print(f"--- Summary of {file_path} ---")
    print(summary[0]['summary_text'])
    return summary[0]['summary_text']

@app.local_entrypoint()
def main():
    # This runs on your local machine and triggers the cloud function
    describe_code.remote(__file__)
