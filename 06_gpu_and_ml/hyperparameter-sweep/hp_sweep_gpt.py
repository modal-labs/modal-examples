# ---
# deploy: true
# args: ["--n-steps", "200", "--n-steps-before-checkpoint", "50", "--n-steps-before-eval", "50"]
# ---

# # Train an SLM from scratch with early-stopping grid search over hyperparameters

# ![shakespeare](./shakespeare.png)

# When you want a language model that performs well on your task, there are three options,
# ordered by the degree of customization:

# - [**Prompt Engineering**](https://en.wikipedia.org/wiki/Prompt_engineering):
# large and capable language models understand tasks in natural language, so you can
# carefully design a natural language "prompt" to elicit the desired behavior.

# - [**Fine-Tuning**](https://modal.com/docs/examples/llm-finetuning):
# those same language models were trained by gradient descent on data sets representing tasks,
# and they can be further trained by gradient descent on data sets representative of your task.

# - **Training from Scratch**:
# if you have enough data for your task, you can throw the pretrained model away and make your own.

# Each step adds additional engineering complexity, but also leads to a superior cost-performance Pareto frontier
# for your tasks. Fine-tuned models at one-tenth the size regularly outperform more generic models,
# and models trained from scratch outperform them.

# Because these models are so much smaller than the Large Language Models that power generic
# assistant chatbots like ChatGPT and Claude, they are often called _Small Language Models_ (SLMs).

# In this example, we will explore training an SLM from scratch on Modal.

# In fact, we'll train 8 SLMs in parallel with different hyperparameters
# and then select the best one for additional training.

# We'll monitor this training live and serve our training and trained models
# as web endpoints and simple browser UIs.

# Along the way we'll use many features of the Modal platform:
# [distributed volumes](https://modal.com/docs/guide/volumes),
# multiple [web endpoints](https://modal.com/docs/guide/webhooks),
# and [parallel container execution](https://modal.com/docs/guide/scale#parallel-execution-of-inputs).

# Together, these features give every machine learning and AI team
# the same infrastructural capabilities that the most sophisticated companies
# have in their internal platforms.

# ## Training

# ### Basic Setup

import logging as L
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import modal
from fastapi import FastAPI
from fastapi.responses import FileResponse
from modal import Image
from pydantic import BaseModel

MINUTES = 60  # seconds

# We'll use A10G GPUs for training which are able to train the model to recognizably improved performance
# in ~15 minutes while keeping costs under ~$1.

gpu = "A10G"

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
        "tensorboard==2.17.1",
        "pydantic>=2",
        "fastapi==0.114.2",
        "numpy<2",
    )
    .copy_local_file(Path(__file__).parent / "model.py", "/root/model.py")
)

monitoring_image = Image.debian_slim(python_version="3.11").pip_install(
    "tensorboard==2.17.1"
)

ui_image = Image.debian_slim(python_version="3.11").pip_install(
    "gradio~=4.44.0", "pydantic>=2", "fastapi==0.114.2"
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

# Training consists of specifying optimization parameters, loading the
# `dataset`, building the `model`, setting up TensorBoard logging &
# checkpointing, and then finally the `training_loop` itself.


@app.function(
    image=image, volumes={volume_path: volume}, gpu=gpu, timeout=20 * MINUTES
)
def train_model(
    node_rank,
    n_nodes,
    hparams,
    experiment_name,
    run_to_first_save=False,
    n_steps=3000,
    n_steps_before_eval=None,
    n_steps_before_checkpoint=None,
):
    # optimizer, data, and model prep
    batch_size = 64
    learning_rate = 3e-4

    n_eval_steps = 100
    n_steps_before_eval = (
        n_steps_before_eval
        if n_steps_before_eval is not None
        else int(n_steps / 8)  # eval eight times per run
    )
    n_steps_before_checkpoint = (
        n_steps_before_checkpoint
        if n_steps_before_checkpoint is not None
        else int(n_steps / 4)  # save four times per run
    )

    train_percent = 0.9

    L.basicConfig(
        level=L.INFO,
        format=f"\033[0;32m%(asctime)s %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] [Node {node_rank+1}/{n_nodes}] %(message)s\033[0m",
        datefmt="%b %d %H:%M:%S",
    )

    # use GPU if available
    device = "cuda"
    L.info("Remote Device: %s // GPU: %s", device, gpu)

    input_file_path = volume_path / "shakespeare_char.txt"
    text = prepare_data(input_file_path, volume)

    # construct dataset
    dataset = Dataset(
        text,
        train_percent,
        batch_size,
        hparams.context_size,
        n_eval_steps,
        device,
    )

    # build model
    model = build_model(hparams, dataset.vocab_size, device)
    num_parameters = sum(p.numel() for p in model.parameters())
    L.info(f"Num parameters: {num_parameters}")

    # setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # TensorBoard logging & checkpointing prep
    model_name = (
        f"{experiment_name}"
        f"_context_size={hparams.context_size}_n_heads={hparams.n_heads}"
        f"_dropout={hparams.dropout}"
    )
    L.info(f"Model Name: {model_name}")

    # save logs to something like:
    # volume/logs/E2024-01-01-000000.000000/
    #   E2024-01-01-000000.000000_context=8_n_heads=1_dropout=0.0/train
    model_log_dir = tb_log_path / f"{experiment_name}/{model_name}"
    model_log_dir.mkdir(parents=True, exist_ok=True)
    train_writer = SummaryWriter(log_dir=f"{model_log_dir}/train")
    val_writer = SummaryWriter(log_dir=f"{model_log_dir}/val")

    # save hyperparameters to TensorBoard for easy reference
    pretty_hparams_str = "\n".join(
        f"{k}: {v}" for k, v in hparams.__dict__.items()
    )
    pretty_hparams_str += f"\nNum parameters: {num_parameters}"
    train_writer.add_text("Hyperparameters", pretty_hparams_str)

    model_save_dir = save_path / experiment_name / model_name
    if model_save_dir.exists():
        L.info("Loading model from checkpoint...")
        checkpoint = torch.load(str(model_save_dir / model_filename))
        if not run_to_first_save:
            # create symlink to the best model so it's easy to find for web serving
            os.symlink(
                str(model_save_dir / model_filename),
                str(save_path / experiment_name / best_model_filename),
            )
            volume.commit()  # commit the symlink

        model.load_state_dict(checkpoint["model"])
        start_step = checkpoint["steps"] + 1
    else:
        model_save_dir.mkdir(parents=True, exist_ok=True)
        start_step = 0
        # save metadata for training restarts and inference
        checkpoint = {
            "model": model.state_dict(),
            "chars": dataset.chars,
            "optimizer": optimizer.state_dict(),
            "val_loss": float("inf"),
            "steps": start_step,
            "hparams": hparams,
            "finished_training": False,
        }

    checkpoint_path = model_save_dir / model_filename

    out = training_loop(
        start_step,
        n_steps,
        n_steps_before_eval,
        n_steps_before_checkpoint,
        dataset,
        model,
        optimizer,
        train_writer,
        val_writer,
        checkpoint,
        checkpoint_path,
        run_to_first_save,
    )

    return node_rank, float(out["val"]), hparams


# ### Main Entry Point

# The main entry point coordinates the hyperparameter optimization training.
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

# Training for each model until the first checkpoint, and then stop early so we
# can compare the validation losses. Then we'll restart training for the best
# model and save it to the models directory.


@app.local_entrypoint()
def main(
    n_steps: int = 3000,
    n_steps_before_checkpoint: int = None,
    n_steps_before_eval: int = None,
):
    from datetime import datetime
    from itertools import product

    experiment_name = f"E{datetime.now().strftime('%Y-%m-%d-%H%M%S.%f')}"
    default_hparams = ModelHyperparameters()

    # build list of hyperparameters to train & validate
    nheads_options = (1, default_hparams.n_heads)
    context_size_options = (8, default_hparams.context_size)
    dropout_options = (0.1, default_hparams.dropout)

    hparams_list = [
        ModelHyperparameters(n_heads=h, context_size=c, dropout=d)
        for h, c, d in product(
            nheads_options, context_size_options, dropout_options
        )
    ]

    # run training for each hyperparameter setting
    results = []
    stop_early = True  # stop early so we can compare val losses
    print(f"Testing {len(hparams_list)} hyperparameter settings")
    n_nodes = len(hparams_list)
    for result in train_model.starmap(
        [
            (
                i,
                n_nodes,
                h,
                experiment_name,
                stop_early,
                n_steps,
                n_steps_before_eval,
                n_steps_before_checkpoint,
            )
            for i, h in enumerate(hparams_list)
        ],
        order_outputs=False,
    ):
        # result = (node_rank, val_loss, hparams)
        node_rank = result[0]
        results.append(result)
        print(
            f"[Node {node_rank+1}/{n_nodes}] Finished."
            f" Early stop val loss result: {result[1:]}"
        )

    # find the model and hparams with the lowest validation loss
    best_result = min(results, key=lambda x: x[1])
    print(f"Best early stop val loss result: {best_result}")
    best_hparams = best_result[-1]

    # finish training with best hparams
    node_rank = 0
    n_nodes = 1  # only one node for final training run
    train_model.remote(
        node_rank,
        n_nodes,
        best_hparams,
        experiment_name,
        not stop_early,
        n_steps,
        n_steps_before_eval,
        n_steps_before_checkpoint,
    )


# After running `modal run hp_sweep_gpt::main` you should see output like this:
# ```
# Sep 16 21:20:39 INFO [hp_sweep_gpt.py.train_model:127] [Node 1/8]  Remote Device: cuda // GPU: A10G
# Sep 16 21:20:40 INFO [hp_sweep_gpt.py.train_model:149] [Node 1/8]  Num parameters: 10693697
# Sep 16 21:20:40 INFO [hp_sweep_gpt.py.train_model:156] [Node 1/8]  Model Name: E2024-0916-142031.618259_context_size=8_n_heads=1_dropout=0.1
# Sep 16 21:20:41 INFO [hp_sweep_gpt.py.train_model:225] [Node 1/8]      0) //  1.03s // Train Loss: 3.58 // Val Loss: 3.60
# Sep 16 21:20:41 INFO [hp_sweep_gpt.py.train_model:127] [Node 2/8]  Remote Device: cuda // GPU: A10G
# ...
# ```


# ### Monitoring with TensorBoard

# To monitor our training we will create a TensorBoard WSGI web app, it will
# display the progress of our training across all 8 models. We'll use the latest
# experiment TensorBoard logs available on the `volume`.


@app.function(image=monitoring_image, volumes={volume_path: volume})
@modal.wsgi_app()
def monitor_training():
    import time

    print("📈 TensorBoard: Waiting for logs...")
    ct = 0
    while not tb_log_path.exists():
        ct += 1
        if ct > 10:
            raise Exception("No logs found after 10 seconds.")
        volume.reload()  # make sure we have the latest data.
        time.sleep(1)

    # obtain the latest log path
    tb_log_paths = glob.glob(f"{tb_log_path}/*")
    latest_tb_log_path = max(tb_log_paths, key=os.path.getctime)
    monitor_path = Path(latest_tb_log_path)
    print(f"📈 Monitoring: {monitor_path.name}")

    # start TensorBoard with the latest log path
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


# After training your TensorBoard will look something like this:

# [[./tensorboard.png|alt=TensorBoard]]

# Notice that there are 8 models training, and the one with the lowest
# validation loss at step 600 continues training to 3000 steps.

# ## Serving the trained model as a web endpoint

# ### Setup

# Initialize some variables for web serving:

web_app = FastAPI()
assets_path = Path(__file__).parent / "assets"

# ### Inference class

# Now we will create a class for running inference only on the trained model.

# We choose the latest experiment that has a best model checkpoint
# and to load that model for inference. In case training is still ongoing,
# we check for updated models on the fly and load them if available.


@app.cls(image=image, volumes={volume_path: volume}, gpu=gpu)
class ModelInference:
    def build_encode_decode(self, chars):
        # create funcs for converting text into digits (encode) and
        # vice versa (decode)
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for i, c in enumerate(chars)}

        def encode(s):
            return [stoi[c] for c in s]

        def decode(l):
            return [itos[i] for i in l]

        return encode, decode

    def load_model_impl(self):
        # loop through all model dirs and load the latest available model
        save_model_dirs = glob.glob(f"{save_path}/*")
        sorted_model_dirs = sorted(
            save_model_dirs, key=os.path.getctime, reverse=True
        )
        found_model = False
        for latest_model_dir in sorted_model_dirs:
            if self.use_model_dir == latest_model_dir and self.is_fully_trained:
                return  # Already loaded
            print(f"Attemping to load from: {latest_model_dir} ...")
            try:
                checkpoint = torch.load(
                    f"{latest_model_dir}/{best_model_filename}"
                )
                print("Successfully loaded model.")
                found_model = True
                break
            except Exception as e:
                L.warning(f"Error loading model: {e}")
        if not found_model:
            raise Exception("No models ready for serving.")

        self.use_model_dir = latest_model_dir
        hparams = checkpoint["hparams"]
        chars = checkpoint["chars"]
        steps = checkpoint["steps"]
        val_loss = checkpoint["val_loss"]
        self.is_fully_trained = checkpoint["finished_training"]

        print(
            f"Loaded model with {steps} train steps "
            f" and val loss of {val_loss:.2f}"
            f" (fully_trained={self.is_fully_trained}"
        )

        # reconstruct encode/decode
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
        self.load_model_impl()  # load updated model if aviailable, o/w no op.

        # generate 1000 new characters from input prompt
        n_new_tokens = 1000
        encoded_prompt = self.encode(prompt)
        # create a torch tensor from the encoded prompt
        torch_input = torch.tensor(encoded_prompt, dtype=torch.long)
        torch_input = torch_input.view(1, len(torch_input))  # add batch dim
        torch_input = torch_input.to(self.device)

        # generate new tokens
        gen_out = self.model.generate(torch_input, n_new_tokens)[0]  # 0th batch
        # decode from digits to text
        chars_out = self.decode([x for x in gen_out.tolist()])[
            len(encoded_prompt) :
        ]
        # join the characters into a string and return
        str_out = "".join(chars_out)
        return str_out


# First, we create a simple POST web endpoint for generating text.


class GenerationRequest(BaseModel):
    prompt: str


@app.function()
@modal.web_endpoint(method="POST", docs=True)
def web_generate(request: GenerationRequest):
    output = ModelInference().generate.remote(request.prompt)
    return {"output": output}


# That will allow us to generate text via a simple `curl` command likthis:

# ```bash
# curl -X POST -H 'Content-Type: application/json' --data-binary '{"prompt": "\n"}' https://your-workspace-name--modal-nano-gpt-web-generate-dev.modal.run
# ```

# which will return something like:

# ```json
# {
# "output":
#    "BRUTUS:
#     The broy trefore anny pleasory to
#     wip me state of villoor so:
#     Fortols listhey for brother beat the else
#     Be all, ill of lo-love in igham;
#     Ah, here all that queen and hould you father offer"
# }
# ```

# It's not exactly Shakespeare, but at least it shows our model learned something!

# ### Serving a Gradio UI

# Second, we create a Gradio web UI for generating text via a graphical user interface in the browser.
# That way our fellow team members and stakeholders can easily interact with the model and give feedback.


@app.function(
    image=ui_image,
    concurrency_limit=1,
    allow_concurrent_inputs=1000,
    mounts=[modal.Mount.from_local_dir(assets_path, remote_path="/assets")],
)
@modal.asgi_app()
def ui():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    # call out to the inference in a separate Modal environment with a GPU
    def go(text=""):
        if not text:
            text = "\n"
        return text + ModelInference().generate.remote(text)

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

    # add a Gradio UI around inference
    with gr.Blocks(theme=theme, css=css, title="Tiny LLM") as interface:
        # title
        gr.Markdown(
            "# Generate Shakespeare text using the prompt",
        )

        # input and output
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

        # button to trigger inference and a link to Modal
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

        # example prompts
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


# The Gradio UI will look something like this:
# [[./gradio.png|alt=gradio]]

# ## Addenda

# The remainder of this code is boilerplate for the training loop.
# There's a lot! If you'd rather not write this stuff yourself,
# consider a training framework like [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable)
# or [Hugging Face](https://huggingface.co/transformers/main_classes/trainer.html).


def log_evals(result, step, t_last, val_writer, train_writer):
    runtime_s = timer() - t_last
    L.info(
        f"{step:5d}) // {runtime_s:>5.2f}s"
        f" // Train Loss: {result['train']:.2f} // Val Loss:"
        f" {result['val']:.2f}"
    )
    val_writer.add_scalar("Cross Entropy Loss", result["val"], step)
    val_writer.add_text("Sample Output", result["sample"], step)
    train_writer.flush()

    return result


def training_loop(
    start_step,
    n_steps,
    n_steps_before_eval,
    n_steps_before_checkpoint,
    dataset,
    model,
    optimizer,
    train_writer,
    val_writer,
    checkpoint,
    checkpoint_path,
    run_to_first_save,
):
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
            log_evals(out, step, t_last, val_writer, train_writer)
            volume.commit()
            t_last = timer()

        # save model with checkpoint information
        if step > 0 and step % n_steps_before_checkpoint == 0:
            checkpoint["steps"] = step
            checkpoint["val_loss"] = out["val"]

            # mark as finished if we hit n steps.
            checkpoint["finished_training"] = step >= n_steps

            L.info(f"Saving checkpoint to {checkpoint_path}")
            save_checkpoint(checkpoint, checkpoint_path)

            if run_to_first_save:
                L.info("Stopping early...")
                break
    return out


def save_checkpoint(checkpoint, checkpoint_path):
    torch.save(checkpoint, checkpoint_path)
    volume.commit()


def prepare_data(input_file_path: Path, volume: modal.Volume) -> str:
    """Download and read the dataset."""
    volume.reload()
    if not input_file_path.exists():
        L.info("Downloading Shakespeare dataset...")
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(data_url, input_file_path)
        volume.commit()
    return input_file_path.read_text()


def build_model(hparams, vocab_size, device):
    """Initialize the model and move it to the device."""
    model = AttentionModel(vocab_size, hparams, device)
    model.to(device)
    return model


def setup_optimizer(model, learning_rate):
    """Set up the optimizer for the model."""
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)
