# ---
# cmd: ["modal", "run", "--detach", "06_gpu_and_ml/long-training.py"]
# mypy: ignore-errors
# ---

# # Run long, resumable training jobs on Modal

# Individual Modal Function calls have a [maximum timeout of 24 hours](https://modal.com/docs/guide/timeouts).
# You can still run long training jobs on Modal by making them interruptible and resumable
# (aka [_reentrant_](https://en.wikipedia.org/wiki/Reentrancy_%28computing%29)).

# This is usually done via checkpointing: saving the model state to disk at regular intervals.
# We recommend implementing checkpointing logic regardless of the duration of your training jobs.
# This prevents loss of progress in case of interruptions or [preemptions](https://modal.com/docs/guide/preemption).

# In this example, we'll walk through how to implement this pattern in
# [PyTorch Lightning](https://lightning.ai/docs/pytorch/2.4.0/).

# But the fundamental pattern is simple and can be applied to any training framework:

# 1. Periodically save checkpoints to a Modal [Volume](https://modal.com/docs/guide/volumes)
# 2. When your training function starts, check the Volume for the latest checkpoint
# 3. Add [retries](https://modal.com/docs/guide/retries) to your training function

# ## Resuming from checkpoints in a training loop

# The `train` function below shows some very simple training logic
# using the built-in checkpointing features of PyTorch Lightning.

# Lightning uses a special filename, `last.ckpt`,
# to indicate which checkpoint is the most recent.
# We check for this file and resume training from it if it exists.

from pathlib import Path
from typing import Optional

import modal


def train(experiment):
    experiment_dir = CHECKPOINTS_PATH / experiment
    last_checkpoint = experiment_dir / "last.ckpt"

    if last_checkpoint.exists():
        print(f"⚡️ resuming training from the latest checkpoint: {last_checkpoint}")
        train_model(
            DATA_PATH,
            experiment_dir,
            resume_from_checkpoint=last_checkpoint,
        )
        print("⚡️ training finished successfully")
    else:
        print("⚡️ starting training from scratch")
        train_model(DATA_PATH, experiment_dir)


# This implementation works fine in a local environment.
# Running it serverlessly and durably on Modal -- with access to auto-scaling cloud GPU infrastructure
# -- does not require any adjustments to the code.
# We just need to ensure that data and checkpoints are saved in Modal _Volumes_.

# ## Modal Volumes are distributed file systems

# Modal [Volumes](https://modal.com/docs/guide/volumes) are distributed file systems --
# you can read and write files from them just like local disks,
# but they are accessible to all of your Modal Functions.
# Their performance is tuned for [Write-Once, Read-Many](https://en.wikipedia.org/wiki/Write_once_read_many) workloads
# with small numbers of large files.

# You can attach them to any Modal Function that needs access.

# But first, you need to create them:

volume = modal.Volume.from_name("example-long-training", create_if_missing=True)

# ## Porting training to Modal

# To attach a Modal Volume to our training function, we need to port it over to run on Modal.

# That means we need to define our training function's dependencies
# (as a [container image](https://modal.com/docs/guide/custom-container))
# and attach it to an application (a [`modal.App`](https://modal.com/docs/guide/apps)).

# Modal Functions that run on GPUs [already have CUDA drivers installed](https://modal.com/docs/guide/cuda),
# so dependency specification is straightforward.
# We just `pip_install` PyTorch and PyTorch Lightning.

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "lightning~=2.4.0", "torch~=2.4.0", "torchvision==0.19.0"
)

app = modal.App("example-long-training", image=image)

# Next, we attach our training function to this app with `app.function`.

# We define all of the serverless infrastructure-specific details of our training at this point.
# For resumable training, there are three key pieces: attaching volumes, adding retries, and setting the timeout.

# We want to attach the Volume to our Function so that the data and checkpoints are saved into it.
# In this sample code, we set these paths via global variables, but in another setting,
# these might be set via environment variables or other configuration mechanisms.

volume_path = Path("/experiments")
DATA_PATH = volume_path / "data"
CHECKPOINTS_PATH = volume_path / "checkpoints"

volumes = {volume_path: volume}

# Then, we define how we want to restart our training in case of interruption.
# We can use `modal.Retries` to add automatic retries to our Function.
# We set the delay time to `0.0` seconds, because on pre-emption or timeout we want to restart immediately.
# We set `max_retries` to the current maximum, which is `10`.

retries = modal.Retries(initial_delay=0.0, max_retries=10)

# Timeouts on Modal are set in seconds, with a minimum of 10 seconds and a maximum of 24 hours.
# When running training jobs that last up to week, we'd set that timeout to 24 hours,
# which would give our training job a maximum of 10 days to complete before we'd need to manually restart.

# For this example, we'll set it to 30 seconds. When running the example, you should observe a few interruptions.

timeout = 30  # seconds

# Now, we put all of this together by wrapping `train` and decorating it
# with `app.function` to add all the infrastructure. We add `max_inputs=1` to ensure that our retries
# will always kickoff in a fresh container.


@app.function(
    volumes=volumes, gpu="a10g", timeout=timeout, retries=retries, max_inputs=1
)
def train_interruptible(*args, **kwargs):
    train(*args, **kwargs)


# ## Kicking off interruptible training

# We define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps)
# to kick off the training job from the local Python environment.


@app.local_entrypoint()
def main(experiment: Optional[str] = None):
    if experiment is None:
        from uuid import uuid4

        experiment = uuid4().hex[:8]
    print(f"⚡️ starting interruptible training experiment {experiment}")
    train_interruptible.spawn(experiment).get()


# It's important to use `.spawn(...).get()` because `.remote` created Function Calls
# expire after 24 hours.

# You can run this with
# ```bash
# modal run --detach 06_gpu_and_ml/long-training.py
# ```

# You should see the training job start and then be interrupted,
# producing a large stack trace in the terminal in red font.
# The job will restart within a few seconds.

# The `--detach` flag ensures training will continue even if you close your terminal or turn off your computer.
# Try detaching and then watch the logs in the [Modal dashboard](https://modal.com/apps).


# ## Details of PyTorch Lightning implementation

# This basic pattern works for any training framework or for custom training jobs --
# or for any reentrant work that can save state to disk.

# But to make the example complete, we include all the details of the PyTorch Lightning implementation below.

# PyTorch Lightning offers [built-in checkpointing](https://pytorch-lightning.readthedocs.io/en/1.2.10/common/weights_loading.html).
# You can specify the checkpoint file path that you want to resume from using the `ckpt_path` parameter of
# [`trainer.fit`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html)
# Additionally, you can specify the checkpointing interval with the `every_n_epochs` parameter of
# [`ModelCheckpoint`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html).


def get_checkpoint(checkpoint_dir):
    from lightning.pytorch.callbacks import ModelCheckpoint

    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=True,
        every_n_epochs=10,
        filename="{epoch:02d}",
    )


def train_model(data_dir, checkpoint_dir, resume_from_checkpoint=None):
    import lightning as L

    autoencoder = get_autoencoder()
    train_loader = get_train_loader(data_dir=data_dir)
    checkpoint_callback = get_checkpoint(checkpoint_dir)

    trainer = L.Trainer(
        limit_train_batches=100, max_epochs=100, callbacks=[checkpoint_callback]
    )
    if resume_from_checkpoint is not None:
        trainer.fit(
            model=autoencoder,
            train_dataloaders=train_loader,
            ckpt_path=resume_from_checkpoint,
        )
    else:
        trainer.fit(autoencoder, train_loader)


def get_autoencoder(checkpoint_path=None):
    import lightning as L
    from torch import nn, optim

    class LitAutoEncoder(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3)
            )
            self.decoder = nn.Sequential(
                nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28)
            )

        def training_step(self, batch, batch_idx):
            x, _ = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss = nn.functional.mse_loss(x_hat, x)
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

    return LitAutoEncoder()


def get_train_loader(data_dir):
    from torch import utils
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor

    print("⚡ setting up data")
    dataset = MNIST(data_dir, download=True, transform=ToTensor())
    train_loader = utils.data.DataLoader(dataset, num_workers=4)
    return train_loader
