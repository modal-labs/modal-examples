# # Llama 2 inference with MLC
#
# [Machine Learning Compilation (MLC)](https://mlc.ai/mlc-llm/) is high-performance tool for serving
# LLMs including Llama 2. We will use the `mlc_chat` and the pre-compiled
# Llama 2 binaries to run inference using a Modal GPU.
#
# This example is adapted from this [MLC chat collab](https://colab.research.google.com/github/mlc-ai/notebooks/blob/main/mlc-llm/tutorial_chat_module_getting_started.ipynb#scrollTo=yYwjsCOK7Jij).
import time
import modal
import queue
import threading

from typing import Generator, List, Dict

# Determine which [GPU](https://modal.com/docs/guide/gpu#gpu-acceleration) you want to use.
GPU: str = "a10g"

# Chose model size. At the time of writing MLC chat only
# provides compiled binaries for Llama 7b and 13b.
LLAMA_MODEL_SIZE: str = "13b"

# Define the image and [Modal Stub](https://modal.com/docs/reference/modal.Stub#modalstub).
# We use an [official NVIDIA CUDA 12.1 image](https://hub.docker.com/r/nvidia/cuda)
# to match MLC CUDA requirements.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04",
        setup_dockerfile_commands=[
            "RUN apt-get update",
            "RUN apt-get install -y python3 python3-pip python-is-python3 git curl",
            # Install git lfs
            "RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash",
            "RUN apt install git-lfs -y",
        ],
    ).run_commands(
        "pip3 install --pre --force-reinstall mlc-ai-nightly-cu121 mlc-chat-nightly-cu121 -f https://mlc.ai/wheels"
    )
    # "These commands will download many prebuilt libraries as well as the chat
    # configuration for Llama-2-7b that mlc_chat needs" [...]
    .run_commands(
        "mkdir -p dist/prebuilt",
        "git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt/lib",
        f"cd dist/prebuilt && git clone https://huggingface.co/mlc-ai/mlc-chat-Llama-2-{LLAMA_MODEL_SIZE}-chat-hf-q4f16_1",
    )
)
stub = modal.Stub("mlc-inference", image=image)


LOADING_MESSAGE: str = f"""

                      #%%%%%%%%%%%%(         #%%%%%%%%%%%%#
                    ,%##%,         %%      .%#(%/         %%.
                   %%.  .%#         (%*   (%*   %%         *%/
                 .%#      %%.        .%% %%      (%*         %%
                #%,        ,%%%%%%%%%%%%%/        .%%         (%*
               %%         (%*         %%*%(         #%,         %%
             (%*         %%         *%(   %%         .%#         #%,
            %%         *%/         %%.     *%(         #%,        .%#
          /%(         %%         .%#         %%         .%%%%%%%%%%%%%.
           (%/      ,%#         #%,           /%(      .%%         (%,
             %%    %%.        .%%               %%    (%*         %%
              (%*.%#         (%*                 /%/ %%         /%/
                %%%%%%%%%%%%%%                     %%%%%%%%%%%%%%

                      LOADING => Llama 2 ({LLAMA_MODEL_SIZE}) [{GPU}]


"""


# The `generate` function will load MLC chat and the compiled model into
# memory and run inference on an input prompt. This is a generator, streaming
# tokens back to the client as they are generated.
@stub.function(gpu=GPU)
def generate(prompt: str) -> Generator[Dict[str, str], None, None]:
    from mlc_chat import ChatModule
    from mlc_chat.callback import DeltaCallback

    yield {
        "type": "loading",
        "message": LOADING_MESSAGE + "\n\n",
    }

    class QueueCallback(DeltaCallback):
        """Stream the output of the chat module to client."""

        def __init__(self, callback_interval: float):
            super().__init__()
            self.queue: queue.Queue = queue.Queue()
            self.stopped = False
            self.callback_interval = callback_interval

        def delta_callback(self, delta_message: str):
            self.stopped = False
            self.queue.put(delta_message)

        def stopped_callback(self):
            self.stopped = True

    cm = ChatModule(
        model=f"/dist/prebuilt/mlc-chat-Llama-2-{LLAMA_MODEL_SIZE}-chat-hf-q4f16_1",
        lib_path=f"/dist/prebuilt/lib/Llama-2-{LLAMA_MODEL_SIZE}-chat-hf-q4f16_1-cuda.so",
    )
    queue_callback = QueueCallback(callback_interval=1)

    # Generate tokens in a background thread so we can yield tokens
    # to caller as a generator.
    def _generate():
        cm.generate(
            prompt=prompt,
            progress_callback=queue_callback,
        )

    background_thread = threading.Thread(target=_generate)
    background_thread.start()

    # Yield as a generator to caller function and spawn
    # text-to-speech functions.
    while not queue_callback.stopped:
        yield {"type": "output", "message": queue_callback.queue.get()}


# ## Run the model
# Create a local Modal entrypoint that call sthe `generate` function.
# This uses the `curses` to render tokens as they are streamed back
# from Modal.
#
# Run this locally with `modal run -q mlc_inference.py`
@stub.local_entrypoint()
def main(prompt: str):
    import curses

    def _generate(stdscr):
        buffer: List[str] = []

        def _buffered_message():
            return "".join(buffer) + ("\n" * 4)

        start = time.time()
        for payload in generate.remote_gen(prompt):
            message = payload["message"]
            if payload["type"] == "loading":
                stdscr.clear()
                stdscr.addstr(0, 0, message)
                stdscr.refresh()
            else:
                buffer.append(message)
                stdscr.clear()
                stdscr.addstr(0, 0, _buffered_message())
                stdscr.refresh()

        n_tokens = len(buffer)
        elapsed = time.time() - start
        print(
            f"[DONE] {n_tokens} tokens generated in {elapsed:.2f}s ({n_tokens / elapsed:.0f} tok/s). Press any key to exit."
        )
        stdscr.getch()

    curses.wrapper(_generate)
