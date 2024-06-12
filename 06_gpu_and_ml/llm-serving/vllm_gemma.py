# # Fast inference with vLLM (Gemma 7B)
#
# In this example, we show how to run basic LLM inference, using [`vLLM`](https://github.com/vllm-project/vllm)
# to take advantage of [PagedAttention](https://arxiv.org/abs/2309.06180), which speeds up inference on longer sequences with optimized key-value caching.
# You can read more about PagedAttention [here](https://charlesfrye.github.io/programming/2023/11/10/llms-systems.html).
#
# We'll run the [Gemma 7B Instruct](https://huggingface.co/google/gemma-7b-it) large language model.
# Gemma is the weights-available version of Google's Gemini model series.
#
# The "7B" in the name refers to the number of parameters (floating point numbers used to control inference)
# in the model. Applying those 7,000,000,000 numbers onto an input is a lot of work,
# so we'll use a GPU to speed up the process -- specifically, a top-of-the-line [NVIDIA H100](https://modal.com/blog/introducing-h100).
#
# "Instruct" means that this version of Gemma is not simply a statistical model of language,
# but has been fine-tuned to follow instructions -- like ChatGPT or Claude,
# it is a model of an assistant that can understand and follow instructions.
#
# You can expect cold starts in under 30 seconds and well over 1000 tokens/second throughput.
# The larger the batch of prompts, the higher the throughput. For example, with the 64 prompts below,
# we can produce nearly 15k tokens with a latency just over 5 seconds, for a throughput of >2.5k tokens/second.
# That's a lot of text!
#
#
# To run
# [any of the other supported models](https://vllm.readthedocs.io/en/latest/models/supported_models.html),
# just change the model name. You may also need to change engine configuration, like `trust_remote_code`,
# or GPU configuration, in order to run some models.
#
# ## Setup
#
# First we import the components we need from `modal`.

import os
import time

import modal

MODEL_DIR = "/model"
MODEL_NAME = "google/gemma-7b-it"


# ## Define a container image
#
# We want to create a Modal image which has the model weights pre-saved to a directory. The benefit of this
# is that the container no longer has to re-download the model from Hugging Face - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# ### Download the weights
# Make sure you have created a [HuggingFace access token](https://huggingface.co/settings/tokens).
# To access the token in a Modal function, we can create a secret on the [secrets page](https://modal.com/secrets).
# Now the token will be available via the environment variable named `HF_TOKEN`. Functions that inject this secret
# will have access to the environment variable.
#
# We can download the model to a particular directory using the HuggingFace utility function `snapshot_download`.
#
# You may need to accept the license agreement from an account associated with that Hugging Face Token
# to download the model.
def download_model_to_image(model_dir, model_name):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        token=os.environ["HF_TOKEN"],
        ignore_patterns=["*.pt", "*.gguf"],  # Using safetensors
    )
    move_cache()


# ### Image definition
# We’ll start from a Docker Hub image by NVIDIA and install `vLLM`.
# Then we’ll use `run_function` to execute `download_model_to_image`
# and save the resulting files to the container image -- that way we don't need
# to redownload the weights every time we change the server's code or start up more instances of the server.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm==0.4.0.post1",
        "torch==2.1.2",
        "transformers==4.39.3",
        "ray==2.10.0",
        "huggingface_hub==0.19.4",
        "hf-transfer==0.1.4",
    )
    # Use the barebones hf-transfer package for maximum download speeds. Varies from 100MB/s to 1.5 GB/s,
    # so download times can vary from under a minute to tens of minutes.
    # If your download slows down or times out, try interrupting and restarting.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        secrets=[modal.Secret.from_name("huggingface-secret")],
        timeout=60 * 20,
        kwargs={"model_dir": MODEL_DIR, "model_name": MODEL_NAME},
    )
)

app = modal.App(f"example-vllm-{MODEL_NAME}", image=image)

# Using `image.imports` allows us to have a reference to vLLM in global scope without getting an error when our script executes locally.
with image.imports():
    import vllm

# ## Encapulate the model in a class
#
# The inference function is best represented with Modal's [class syntax](/docs/guide/lifecycle-functions) and the `@enter` decorator.
# This enables us to load the model into memory just once every time a container starts up, and keep it cached
# on the GPU for each subsequent invocation of the function.
#
# The `vLLM` library allows the code to remain quite clean!

GPU_CONFIG = modal.gpu.H100(count=1)


@app.cls(gpu=GPU_CONFIG, secrets=[modal.Secret.from_name("huggingface-secret")])
class Model:
    @modal.enter()
    def load(self):
        self.template = (
            "<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
        )

        # Load the model. Tip: Some models, like MPT, may require `trust_remote_code=true`.
        self.llm = vllm.LLM(
            MODEL_DIR,
            enforce_eager=True,  # skip graph capturing for faster cold starts
            tensor_parallel_size=GPU_CONFIG.count,
        )

    @modal.method()
    def generate(self, user_questions):
        prompts = [self.template.format(user=q) for q in user_questions]

        sampling_params = vllm.SamplingParams(
            temperature=0.75,
            top_p=0.99,
            max_tokens=256,
            presence_penalty=1.15,
        )
        start = time.monotonic_ns()
        result = self.llm.generate(prompts, sampling_params)
        duration_s = (time.monotonic_ns() - start) / 1e9
        num_tokens = 0

        COLOR = {
            "HEADER": "\033[95m",
            "BLUE": "\033[94m",
            "GREEN": "\033[92m",
            "RED": "\033[91m",
            "ENDC": "\033[0m",
        }

        for output in result:
            num_tokens += len(output.outputs[0].token_ids)
            print(
                f"{COLOR['HEADER']}{COLOR['GREEN']}{output.prompt}",
                f"\n{COLOR['BLUE']}{output.outputs[0].text}",
                "\n\n",
                sep=COLOR["ENDC"],
            )
            time.sleep(0.01)
        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Generated {num_tokens} tokens from {MODEL_NAME} in {duration_s:.1f} seconds,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second on {GPU_CONFIG}.{COLOR['ENDC']}"
        )

    @modal.exit()
    def stop_engine(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()


# ## Run the model
# We define a [`local_entrypoint`](/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. Run it by executing the command `modal run vllm_inference.py`.
#
# The examples below are meant to put the model through its paces, with a variety of questions and prompts.
# We also calculate the throughput and latency we achieve.
@app.local_entrypoint()
def main():
    questions = [
        # Coding questions
        "Implement a Python function to compute the Fibonacci numbers.",
        "Write a Rust function that performs binary exponentiation.",
        "How do I allocate memory in C?",
        "What are the differences between Javascript and Python?",
        "How do I find invalid indices in Postgres?",
        "How can you implement a LRU (Least Recently Used) cache in Python?",
        "What approach would you use to detect and prevent race conditions in a multithreaded application?",
        "Can you explain how a decision tree algorithm works in machine learning?",
        "How would you design a simple key-value store database from scratch?",
        "How do you handle deadlock situations in concurrent programming?",
        "What is the logic behind the A* search algorithm, and where is it used?",
        "How can you design an efficient autocomplete system?",
        "What approach would you take to design a secure session management system in a web application?",
        "How would you handle collision in a hash table?",
        "How can you implement a load balancer for a distributed system?",
        # Literature
        "What is the fable involving a fox and grapes?",
        "Write a story in the style of James Joyce about a trip to the Australian outback in 2083, to see robots in the beautiful desert.",
        "Who does Harry turn into a balloon?",
        "Write a tale about a time-traveling historian who's determined to witness the most significant events in human history.",
        "Describe a day in the life of a secret agent who's also a full-time parent.",
        "Create a story about a detective who can communicate with animals.",
        "What is the most unusual thing about living in a city floating in the clouds?",
        "In a world where dreams are shared, what happens when a nightmare invades a peaceful dream?",
        "Describe the adventure of a lifetime for a group of friends who found a map leading to a parallel universe.",
        "Tell a story about a musician who discovers that their music has magical powers.",
        "In a world where people age backwards, describe the life of a 5-year-old man.",
        "Create a tale about a painter whose artwork comes to life every night.",
        "What happens when a poet's verses start to predict future events?",
        "Imagine a world where books can talk. How does a librarian handle them?",
        "Tell a story about an astronaut who discovered a planet populated by plants.",
        "Describe the journey of a letter traveling through the most sophisticated postal service ever.",
        "Write a tale about a chef whose food can evoke memories from the eater's past.",
        # History
        "What were the major contributing factors to the fall of the Roman Empire?",
        "How did the invention of the printing press revolutionize European society?",
        "What are the effects of quantitative easing?",
        "How did the Greek philosophers influence economic thought in the ancient world?",
        "What were the economic and philosophical factors that led to the fall of the Soviet Union?",
        "How did decolonization in the 20th century change the geopolitical map?",
        "What was the influence of the Khmer Empire on Southeast Asia's history and culture?",
        # Thoughtfulness
        "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
        "In a dystopian future where water is the most valuable commodity, how would society function?",
        "If a scientist discovers immortality, how could this impact society, economy, and the environment?",
        "What could be the potential implications of contact with an advanced alien civilization?",
        "Describe how you would mediate a conflict between two roommates about doing the dishes using techniques of non-violent communication.",
        # Math
        "What is the product of 9 and 8?",
        "If a train travels 120 kilometers in 2 hours, what is its average speed?",
        "Think through this step by step. If the sequence a_n is defined by a_1 = 3, a_2 = 5, and a_n = a_(n-1) + a_(n-2) for n > 2, find a_6.",
        "Think through this step by step. Calculate the sum of an arithmetic series with first term 3, last term 35, and total terms 11.",
        "Think through this step by step. What is the area of a triangle with vertices at the points (1,2), (3,-4), and (-2,5)?",
        "Think through this step by step. Solve the following system of linear equations: 3x + 2y = 14, 5x - y = 15.",
        # Facts
        "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
        "What is the Voynich manuscript, and why has it perplexed scholars for centuries?",
        "What was Project A119 and what were its objectives?",
        "What is the 'Dyatlov Pass incident' and why does it remain a mystery?",
        "What is the 'Emu War' that took place in Australia in the 1930s?",
        "What is the 'Phantom Time Hypothesis' proposed by Heribert Illig?",
        "Who was the 'Green Children of Woolpit' as per 12th-century English legend?",
        "What are 'zombie stars' in the context of astronomy?",
        "Who were the 'Dog-Headed Saint' and the 'Lion-Faced Saint' in medieval Christian traditions?",
        "What is the story of the 'Globsters', unidentified organic masses washed up on the shores?",
        # Multilingual
        "战国时期最重要的人物是谁?",
        "Tuende hatua kwa hatua. Hesabu jumla ya mfululizo wa kihesabu wenye neno la kwanza 2, neno la mwisho 42, na jumla ya maneno 21.",
        "Kannst du die wichtigsten Eigenschaften und Funktionen des NMDA-Rezeptors beschreiben?",
    ]
    model = Model()
    model.generate.remote(questions)
