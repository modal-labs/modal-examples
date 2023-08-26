"""
Adapted from https://colab.research.google.com/github/mlc-ai/notebooks/blob/main/mlc-llm/tutorial_chat_module_getting_started.ipynb#scrollTo=yYwjsCOK7Jij
"""
import modal
import queue
import threading

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
    )
    .run_commands(
            "pip3 install --pre --force-reinstall mlc-ai-nightly-cu121 mlc-chat-nightly-cu121 -f https://mlc.ai/wheels"
    )
    # "These commands will download many prebuilt libraries as well as the chat configuration for Llama-2-7b that mlc_chat needs" [...]
    .run_commands(
        "mkdir -p dist/prebuilt",
        "git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt/lib",
        "cd dist/prebuilt && git clone https://huggingface.co/mlc-ai/mlc-chat-Llama-2-13b-chat-hf-q4f16_1"
    )
)
stub = modal.Stub("lamma2-to-speech", image=image)



@stub.function(gpu=modal.gpu.A10G())
def generate(prompt:str):
    from mlc_chat import ChatModule
    from mlc_chat.callback import DeltaCallback
    
    class QueueCallback(DeltaCallback):
        """Stream the output of the chat module to stdout."""

        def __init__(self, callback_interval: int = 2):
            super().__init__()
            self.queue = queue.Queue()
            self.stopped = False
            self.callback_interval = callback_interval

        def delta_callback(self, delta_message: str):
            self.stopped = False
            self.queue.put(delta_message)

        def stopped_callback(self):
            self.stopped = True

    cm = ChatModule(model="/dist/prebuilt/mlc-chat-Llama-2-13b-chat-hf-q4f16_1", lib_path="/dist/prebuilt/lib/Llama-2-13b-chat-hf-q4f16_1-cuda.so")
    queue_callback = QueueCallback(callback_interval=1)

    # Generate tokens in a background thread.
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
        yield queue_callback.queue.get()

@stub.local_entrypoint()
def main(prompt:str):

    import curses
    def _generate(stdscr):
        buffer = []
        def _display():
            return "".join(buffer) + "\n\n\n\n"

        stdscr.addstr(1, 0, _display())
        stdscr.refresh()

        for message in generate.remote_gen(prompt):
            buffer.append(message)
            stdscr.clear()
            stdscr.addstr(0, 0, _display())
            stdscr.refresh()

        print("[DONE] Hit any key to exit")
        stdscr.getch()

    curses.wrapper(_generate)
