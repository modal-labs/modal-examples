# ---
# deploy: true
# ---
# # LLM Training with Hyperparameter Optimization
#
# <img src="./shakespeare.png" alt='shakespeare'>
#
# When you want an LLM tailored to your specific data there are three options.
# The easiest is [Prompt Engineering](https://en.wikipedia.org/wiki/Prompt_engineering)
# but the quality of the results aren't very high. The next option is
# [fine-tuning](https://modal.com/docs/examples/llm-finetuning) which is more
# involved and improves results significantly.  The final option is training an LLM
# from scratch which is the most involved but may allow for the highest caliber results.
# In addition, you may be able to shrink the model considerably and save money on inference
# costs after training.
#
# In this example we will explore training from scratch. In fact, we'll train
# 8 LLMs in parallel with different hyperparameters and then select the best
# one. Along the way we will utlize many Modal utilities: [distributed volumes](https://modal.com/docs/guide/volumes),
# multiple [web endpoints](https://modal.com/docs/guide/webhooks),
# and [parallel container execution](https://modal.com/docs/guide/scale#parallel-execution-of-inputs),
# in essence showing you how to combine multiple techniques into one powerful project. Sound
# challenging? Modal makes it easy.
#
# ## Training
# ### Basic Setup
# First we `import modal`, `fastapi` for serving tensorboard, torch
# LLM model (`AttentionModel`), and a `Dataset` class. The torch model is a nano GPT style model
# very similar to [Karpathy's](https://github.com/ShariqM/modal_nano_gpt/blob/master/model.py).
# The `Dataset` class manages the Shakespeare text data which is available
# [here](/modal_nano_gpt/blob/master/model.py).

import logging as L

import modal

L.basicConfig(
    level=L.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%b %d %H:%M:%S",
)

import urllib.request
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from modal import Image

# We'll use A10G GPUs for training which are able to train the model
# in 10 minutes while keeping costs under ~$1. Since the default modal function
# [timeout](https://modal.com/docs/guide/timeouts) is only 5 minutes
# we need to increase the 20 minutes.

gpu = "A10G"
timeout_s = 20 * 60  # 20 minutes


# ### Create a Volume
# Since we'll be coordinating training across multiple machines we'll use a
# single [Volume](https://modal.com/docs/guide/volumes)
# to store the `dataset`, checkpointed models, and TensorBoard logs. Modal Volumes do
# not automatically synchronize writes so we'll have to be careful to use
# `commit()` and `reload()` calls when appropriate.

volume = modal.Volume.from_name(
    "example-hp-sweep-gpt-volume", create_if_missing=True
)
volume_path = Path("/vol/data")
model_filename = "nano_gpt_model.pt"
best_model_filename = "best_nano_gpt_model.pt"
tb_log_path = volume_path / "tb_logs"
save_path = volume_path / "models"

# ### Define a container image
# The container image is based on the latest Debian slim image with `torch`
# for training, `gradio` for serving a web interface, and `tensorboard` for
# monitoring training.

image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.2",
        "gradio~=4.44.0",
        "pydantic>=2",
        "tensorboard==2.17.1",
        "fastapi==0.114.2",
        "numpy<2",
    )
    .copy_local_file(Path(__file__).parent / "model.py", "/root/model.py")
)

app = modal.App("example-hp-sweep-gpt")

with image.imports():
    import glob
    import os
    from timeit import default_timer as timer

    import tensorboard
    import torch
    from model import AttentionModel, Dataset
    from torch.utils.tensorboard import SummaryWriter

# ### Training Function

# Here we define the training function making sure to include the `image`,
# `volume`, `gpu`, and `timeout` parameters.

# Training consists of specificying optimization parameters, loading the
# `dataset`, building the `model`, setting up tensorboard logging,
# checkpointing, and finally the training itself.


@app.function(
    image=image, volumes={volume_path: volume}, gpu=gpu, timeout=timeout_s
)
def train_model(
    node_rank, n_nodes, hparams, experiment_name, run_to_first_save=False
):
    # Optimization, Data, and Model prep ###
    batch_size = 64
    n_steps = 3000
    n_eval_steps = 100
    n_steps_before_eval = int(n_steps / 8)  # eval eight times per run
    n_steps_before_checkpoint = int(n_steps / 4)  # save four times per run
    train_percent = 0.9
    learning_rate = 3e-4
    prepend_logs = f"[Node {node_rank+1}/{n_nodes}] "

    # Use GPU if available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L.info(f"{prepend_logs} Remote Device: {device} // GPU: {gpu}")

    input_file_path = volume_path / "shakespeare_char.txt"
    volume.reload()  # Make sure we have the latest data.
    if not os.path.exists(input_file_path):
        L.info(f"{prepend_logs} Downloading Shakespeare dataset...")
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(data_url, input_file_path)

        volume.commit()  # Commit to disk

    # Construct dataset
    with open(input_file_path, "r", encoding="utf-8") as f:
        text = f.read()
    dataset = Dataset(
        text,
        train_percent,
        batch_size,
        hparams.context_size,
        n_eval_steps,
        device,
    )

    # Build Model
    model = AttentionModel(dataset.vocab_size, hparams, device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_parameters = sum(p.numel() for p in model.parameters())
    L.info(f"{prepend_logs} Num parameters: {num_parameters}")

    # Tensorboard logging & checkpointing prep
    model_name = (
        f"{experiment_name}"
        f"_context_size={hparams.context_size}_n_heads={hparams.n_heads}"
        f"_dropout={hparams.dropout}"
    )
    L.info(f"{prepend_logs} Model Name: {model_name}")

    # Save logs to something like:
    # volume/logs/E2024-01-01-000000.000000/
    #   E2024-01-01-000000.000000_context=8_n_heads=1_dropout=0.0/train
    model_log_dir = tb_log_path / f"{experiment_name}/{model_name}"
    os.makedirs(model_log_dir, exist_ok=True)
    train_writer = SummaryWriter(log_dir=f"{model_log_dir}/train")
    val_writer = SummaryWriter(log_dir=f"{model_log_dir}/val")

    # Save hyperparameters to tensorboard for easy reference
    pretty_hparams_str = ""
    for k, v in hparams.__dict__.items():
        pretty_hparams_str += f"{k}: {v}\n"
    pretty_hparams_str += f"Num parameters: {num_parameters}"
    train_writer.add_text("Hyperparameters", pretty_hparams_str)

    # Load & Save models to something like:
    # volume/models/E2024-01-01-000000.000000/
    #  E2024-01-01-000000.000000_context=8_n_heads=1_dropout=0.0/nano_gpt_model.pt
    model_save_dir = save_path / experiment_name / model_name
    if model_save_dir.exists():
        L.info(f"{prepend_logs} Loading model from checkpiont...")
        checkpoint = torch.load(str(model_save_dir / model_filename))
        if run_to_first_save:
            L.info(
                f"{prepend_logs} Already done. Container Restart? Stopping early..."
            )
            return node_rank, checkpoint["val_loss"], hparams
        else:
            # Create symlink to the best model so it's easy to find for web serving.
            os.symlink(
                str(model_save_dir / model_filename),
                str(save_path / experiment_name / best_model_filename),
            )
            volume.commit()  # Commit the symlink.

        model.load_state_dict(checkpoint["model"])
        start_step = checkpoint["steps"] + 1
    else:
        assert run_to_first_save, "should have loaded ckpt"  # can remove later.
        os.makedirs(model_save_dir, exist_ok=True)
        start_step = 0
        # Save metadata for training restarts and inference
        checkpoint = {
            "model": model.state_dict(),
            "chars": dataset.chars,
            "optimizer": optimizer.state_dict(),
            "val_loss": float("inf"),
            "steps": start_step,
            "hparams": hparams,
            "finished_training": False,
        }

    # Training
    t_last = timer()
    for step in range(start_step, n_steps + 1):
        # sample a batch of data
        xb, yb = dataset.get_batch("train")

        # evaluate the loss, calculate & apply gradients
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # log training loss
        train_writer.add_scalar("Cross Entropy Loss", loss.item(), step)

        # evaluate model on validation set
        if step % n_steps_before_eval == 0:
            out = dataset.eval_model(model)
            runtime_s = timer() - t_last
            L.info(
                f"{prepend_logs} {step:5d}) // {runtime_s:>5.2f}s"
                f" // Train Loss: {out['train']:.2f} // Val Loss:"
                f" {out['val']:.2f}"
            )
            val_writer.add_scalar("Cross Entropy Loss", out["val"], step)
            t_last = timer()
            train_writer.flush()
            volume.commit()

        # save model with checkpoint information
        if step > 0 and step % n_steps_before_checkpoint == 0:
            L.info(f"{prepend_logs} Saving model to {model_save_dir}")
            checkpoint["finished_training"] = (
                step >= n_steps
            )  # Mark as finished if we hit n steps.
            checkpoint["steps"] = step
            checkpoint["val_loss"] = out["val"]
            torch.save(checkpoint, model_save_dir / model_filename)
            volume.commit()
            if run_to_first_save:
                L.info(f"{prepend_logs} Stopping early...")
                break

    return node_rank, float(out["val"]), hparams


# ### Main Entry Point
# The main entry point runs coordinates the hyperparameter optimization training.
# First we specify the default hyperparameters for the model, taken from
# [Karpathy's biggest model](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5976s),
# which add up to 10 million total neural network parameters.


@dataclass
class ModelHyperparameters:
    n_heads: int = 6
    n_embed: int = 384
    n_blocks: int = 6
    context_size: int = 256
    dropout: float = 0.2


# Next we define the main entry point which runs the hyperparameter
# optimization. It will train 8 models in parallel across 8 containers, each
# with different hyperparameters, varying the number of heads (`n_heads`), the
# `context_size`
# (called the block size in Karpathy lingo), and the dropout rate (`dropout`). To run in
# parallel we need to use the [starmap function](https://modal.com/docs/guide/scale#parallel-execution-of-inputs).
#
# Training for each model until the first checkpoint, and then stop early so we
# can compare the validation losses. Then we'll restart training for the best
# model and save it to the models directory.


@app.local_entrypoint()
def main():
    from datetime import datetime

    experiment_name = f"E{datetime.now().strftime('%Y-%m%d-%H%M%S.%f')}"
    default_hparams = ModelHyperparameters()

    # Build list of hyperparameters to train & validate
    hparams_list = []
    h_options = (1, default_hparams.n_heads)
    c_options = (8, default_hparams.context_size)
    d_options = (0.1, default_hparams.dropout)

    hparams_list = [
        ModelHyperparameters(
            n_heads=n_heads, context_size=context_size, dropout=dropout
        )
        for n_heads in h_options
        for context_size in c_options
        for dropout in d_options
    ]

    # Run training for each hyperparameter setting
    results = []
    stop_early = True  # stop early so we can compare val losses
    L.info(f"Testing {len(hparams_list)} hyperparameter settings")
    n_nodes = len(hparams_list)
    for result in train_model.starmap(
        [
            (i, n_nodes, h, experiment_name, stop_early)
            for i, h in enumerate(hparams_list)
        ],
        order_outputs=False,
    ):
        # result = (node_rank, val_loss, hparams)
        node_rank = result[0]
        results.append(result)
        L.info(
            f"[Node {node_rank+1}/{n_nodes}] Finished."
            f" Early stop val loss result: {result[1:]}"
        )

    # Find the model and hparams with the lowest validation loss
    best_result = min(results, key=lambda x: x[1])
    L.info(f"Best early stop val loss result: {best_result}")
    best_hparams = best_result[-1]

    # Finish training with best hparams
    node_rank = 0
    n_nodes = 1  # Only one node for final training.
    train_model.remote(node_rank, n_nodes, best_hparams, experiment_name)


# After running `modal run hp_sweep_gpt::main` you should see output like this:
# ```
# Sep 16 21:20:39 INFO [hp_sweep_gpt.py.train_model:127] [Node 1/8]  Remote Device: cuda // GPU: A10G
# Sep 16 21:20:40 INFO [hp_sweep_gpt.py.train_model:149] [Node 1/8]  Num parameters: 10693697
# Sep 16 21:20:40 INFO [hp_sweep_gpt.py.train_model:156] [Node 1/8]  Model Name: E2024-0916-142031.618259_context_size=8_n_heads=1_dropout=0.1
# Sep 16 21:20:41 INFO [hp_sweep_gpt.py.train_model:225] [Node 1/8]      0) //  1.03s // Train Loss: 3.58 // Val Loss: 3.60
# Sep 16 21:20:41 INFO [hp_sweep_gpt.py.train_model:127] [Node 2/8]  Remote Device: cuda // GPU: A10G
# ...
# ```


# ### Bonus: Tensorboard Web App
# To monitor our training we will create a Tensorboard WSGI web app, it will
# display the progress of our training across all 8 models. We'll use the latest
# experiment tensorboard logs available on the `volume`.
@app.function(image=image, volumes={volume_path: volume})
@modal.wsgi_app()
def monitor_training():
    import time

    L.info("Tensorboard: Waiting 10 seconds for training to start...")
    time.sleep(10)  # Wait for experiment folder to be created by training.
    volume.reload()  # Make sure we have the latest data.

    # Obtain the latest log path
    tb_log_paths = glob.glob(f"{tb_log_path}/*")
    latest_tb_log_path = max(tb_log_paths, key=os.path.getctime)
    monitor_path = Path(latest_tb_log_path)
    L.info(f"Monitoring: {monitor_path.name}")

    # Start tensorboard with the latest log path
    board = tensorboard.program.TensorBoard()
    board.configure(logdir=str(monitor_path))
    (data_provider, deprecated_multiplexer) = board._make_data_provider()
    wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
        board.flags,
        board.plugin_loaders,
        data_provider,
        board.assets_zip_provider,
        deprecated_multiplexer,
    )
    return wsgi_app


# After training your Tensorboard will look something like this:
# [[./tensorboard.png|alt=tensorboard]]
# Notice that there are 8 models training, and the one with the lowest
# validation loss at step 600 continues training to 3000 steps.

# ## Web Serving (another bonus)
# ### Setup
# Initialize some variables for web serving:

web_app = FastAPI()
assets_path = Path(__file__).parent / "assets"

# ### Inference class
# Now we will create a class for running inference only on the trained model.
#
# We choose the latest experiment that has a best model checkpoint
# and to load that model for inference. In case training is still ongoing,
# we check for updated models on the fly and load them if available.


@app.cls(image=image, volumes={volume_path: volume}, gpu=gpu)
class ModelInference:
    def build_encode_decode(self, chars):
        # Create funcs for converting  text into digits (encode) and
        # vice versa (decode)
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for i, c in enumerate(chars)}

        def encode(s):
            return [stoi[c] for c in s]

        def decode(l):
            return [itos[i] for i in l]

        return encode, decode

    def load_model_impl(self):
        # Loop through all model dirs and load the latest available model
        save_model_dirs = glob.glob(f"{save_path}/*")
        sorted_model_dirs = sorted(
            save_model_dirs, key=os.path.getctime, reverse=True
        )
        found_model = False
        for latest_model_dir in sorted_model_dirs:
            if self.use_model_dir == latest_model_dir and self.is_fully_trained:
                return  # Already loaded
            L.info(f"Attemping to load from: {latest_model_dir} ...")
            try:
                checkpoint = torch.load(
                    f"{latest_model_dir}/{best_model_filename}"
                )
                L.info("Successfully loaded model.")
                found_model = True
                break
            except Exception as e:
                L.warning(f"Error loading model: {e}")
        if not found_model:
            raise Exception("No models ready for serving.")

        # Model loaded successfully. Print info about the model
        self.use_model_dir = latest_model_dir
        hparams = checkpoint["hparams"]
        chars = checkpoint["chars"]
        steps = checkpoint["steps"]
        val_loss = checkpoint["val_loss"]
        self.is_fully_trained = checkpoint["finished_training"]

        L.info(
            f"Loaded model with {steps} train steps "
            f" and val loss of {val_loss:.2f}"
            f" (fully_trained={self.is_fully_trained}"
        )

        # Reconstruct encode/decode
        vocab_size = len(chars)
        self.encode, self.decode = self.build_encode_decode(chars)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AttentionModel(vocab_size, hparams, self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)

    @modal.enter()
    def load_model(self):
        self.use_model_dir = None
        self.is_fully_trained = False
        self.load_model_impl()

    @modal.method()
    def generate(self, prompt):
        self.load_model_impl()  # Load updated model if aviailable, o/w no op.

        # Generate 1000 new characters from input prompt
        n_new_tokens = 1000
        encoded_prompt = self.encode(prompt)
        # Create a torch tensor from the encoded prompt
        torch_input = torch.tensor(encoded_prompt, dtype=torch.long)
        torch_input = torch_input.view(1, len(torch_input))  # Add batch dim.
        torch_input = torch_input.to(self.device)

        # Generate new tokens
        gen_out = self.model.generate(torch_input, n_new_tokens)[0]  # 0th batch
        # Decode from digits to text
        chars_out = self.decode([x for x in gen_out.tolist()])
        # Join the characters into a string and return
        str_out = "".join(chars_out)
        return str_out


# First, we create a simple POST web endpoint for generating text.


@app.function()
@modal.web_endpoint(method="POST", docs=True)
def web_generate(item: dict):
    output = ModelInference().generate.remote(item["prompt"])
    return {"web_generate": output}


# That will allow us to generate text via a simple `curl` command like this:
# ```bash
# curl -X POST -H 'Content-Type: application/json' --data-binary '{"prompt": "\n"}' https://shariqm--modal-nano-gpt-web-generate-dev.modal.run
# ```
# which will return something like:
# ```bash
# {'web_generate':'\nBRUTUS:\nThe broy trefore anny pleasory to\nwip me state of villoor so:\nFortols listhey for brother beat the else\nBe all, ill of lo-love in igham;\nAh, here all that queen and hould you father offer'}
# ```
#
# It's not exactly Shakespeare, but at least it shows our model learned something!

# Second, we create a Gradio web app for generating text via a nice looking
# website. Notice that we don't include a `gpu` in the `app.function`
# parameters since it's not needed, saving us GPU costs for this container.


@app.function(
    image=image,
    concurrency_limit=1,
    allow_concurrent_inputs=1000,
    mounts=[modal.Mount.from_local_dir(assets_path, remote_path="/assets")],
)
@modal.asgi_app()
def fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    # Call out to the inference in a separate Modal environment with a GPU
    def go(text=""):
        if not text:
            text = "\n"
        return ModelInference().generate.remote(text)

    example_prompts = [
        "DUKE OF YORK:\nWhere art thou Lucas?",
        "ROMEO:\nWhat is a man?",
        "CLARENCE:\nFair is foul and foul is fair, but who are you?",
        "Brevity is the soul of wit, so what is the soul of foolishness?",
    ]

    # custom styles: an icon, a background, and a theme
    @web_app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse("/assets/favicon.svg")

    @web_app.get("/assets/background.svg", include_in_schema=False)
    async def background():
        return FileResponse("/assets/background.svg")

    with open("/assets/index.css") as f:
        css = f.read()

    theme = gr.themes.Default(
        primary_hue="green", secondary_hue="emerald", neutral_hue="neutral"
    )

    # add a gradio UI around inference
    with gr.Blocks(theme=theme, css=css, title="Tiny LLM") as interface:
        # Title
        gr.Markdown(
            "# Generate Shakespeare text using the prompt",
        )

        # Input and Output
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Input:")
                inp = gr.Textbox(  # input text component
                    label="",
                    placeholder="Write some Shakespeare like text or keep it empty!",
                    lines=10,
                )
            with gr.Column():
                gr.Markdown("## Output:")
                out = gr.Textbox(  # output text component
                    label="",
                    lines=10,
                )

        # Button to trigger inference and a link to Modal
        with gr.Row():
            btn = gr.Button("Generate", variant="primary", scale=2)
            btn.click(
                fn=go, inputs=inp, outputs=out
            )  # connect inputs and outputs with inference function

            gr.Button(  # shameless plug
                " Powered by Modal",
                variant="secondary",
                link="https://modal.com",
            )

        # Example prompts
        with gr.Column(variant="compact"):
            # add in a few examples to inspire users
            for ii, prompt in enumerate(example_prompts):
                btn = gr.Button(prompt, variant="secondary")
                btn.click(fn=lambda idx=ii: example_prompts[idx], outputs=inp)

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )


# The Gradio app will look something like this:
# [[./gradio.png|alt=gradio]]

# We hope you enjoyed this example. Message us on Slack if you need help!

# ## Further Examples
# [Scale out](https://modal.com/docs/guide/scale#parallel-execution-of-inputs)
