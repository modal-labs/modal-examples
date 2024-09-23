# ---
# cmd: ["modal", "run", "06_gpu_and_ml.long-training.long-training", "--detach"]
# deploy: true
# ---

# # Running long training jobs on Modal

# While Modal functions typically have a [maximum timeout of 24 hours](/docs/guide/timeouts), you can still run long training jobs on Modal by implementing a checkpointing mechanism in your code.
# This allows you to save the model's state periodically and resume from the last saved state.
# In fact, we recommend implementing checkpointing logic regardless of the duration of your training jobs. This prevents loss of progress in case of interruptions or [preemptions](/docs/guide/preemption).

# In this example, we'll walk through how to implement this pattern using PyTorch Lightning.

# ## Pattern

# The core pattern for long-duration training on Modal:

# 1. Periodically save checkpoints to a Modal [volume](/docs/guide/volumes)
# 2. Handle interruptions/timeouts and resume from the last checkpoint


# ## Setup

# Let's start by importing the Modal client and defining the Modal app and image. Since we are using PyTorch Lightning, we use an officially supported CUDA docker image as our base image.
# Then we install `pytorch` and `lightning` on top of that.

import os

import modal

app = modal.App("interrupt-resume-lightning")

# Set up the environment
image = modal.Image.from_registry(
    "nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10"
).pip_install("lightning", "torchvision")


# ## Define Modal Volume

# Next, we set up a Modal [volume](/docs/guide/volumes) for storing both the training data and the checkpoints

volume = modal.Volume.from_name("training-checkpoints", create_if_missing=True)

VOLUME_PATH = "/vol"
DATA_PATH = f"{VOLUME_PATH}/data"
CHECKPOINTS_PATH = f"{VOLUME_PATH}/checkpoints"

# ## Model training

# We implement the actual model training class/functions and the checkpointing logic.
# PyTorch Lightning offers some [built-in checkpointing](https://pytorch-lightning.readthedocs.io/en/1.2.10/common/weights_loading.html#:~:text=Lightning%20automates%20saving%20and%20loading,having%20to%20retrain%20the%20model.) functionality.
# You can specify the checkpoint file path that you want to resume from using the `ckpt_path` parameter of [`trainer.fit`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html)
# Additionally, you can specify the checkpointing interval with the `every_n_epochs` parameter of [`ModelCheckpoint`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html).
# In the code below, we save checkpoints every 10 epochs, but this number can be adjusted depending on how long the epochs take. The goal is to minimize the disruption from job failures. Something that takes a few days should be checkpointed perhaps every few hours. Depending on what training framework you are using, how exactly this checkpointing gets implemented may vary.


def get_checkpoint(checkpoint_dir):
    from lightning.pytorch.callbacks import ModelCheckpoint

    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=True,
        every_n_epochs=10,
        filename="epoch{epoch:02d}",
    )


def train_model(data_dir, checkpoint_dir, resume_from_checkpoint=None):
    import lightning as L

    from .train import get_autoencoder, get_train_loader

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    autoencoder = get_autoencoder()
    train_loader = get_train_loader(data_dir=data_dir)
    checkpoint_callback = get_checkpoint(checkpoint_dir)
    trainer = L.Trainer(
        limit_train_batches=100, max_epochs=100, callbacks=[checkpoint_callback]
    )
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        trainer.fit(
            model=autoencoder,
            train_dataloaders=train_loader,
            ckpt_path=resume_from_checkpoint,
        )
    else:
        print("Starting training from scratch")
        trainer.fit(autoencoder, train_loader)
    print("Done training")
    return


# ## Training function deployed on Modal
#
# Next, we define the training function running on Modal infrastructure. Note that this function has the volume mounted on it.
# The training function checks in the volume for an existing latest checkpoint file, and resumes training off that checkpoint if it finds it.
# The `timeout` parameter in the `@app.function` decorator is set to 30 seconds for demonstration purposes. In a real scenario, you'd set this to a larger value (e.g., several hours) based on your needs.
@app.function(
    image=image,
    # mounts=[train_script_mount],
    volumes={VOLUME_PATH: volume},
    gpu="any",
    timeout=30,
)
def train():
    last_checkpoint = os.path.join(CHECKPOINTS_PATH, "last.ckpt")

    try:
        if os.path.exists(last_checkpoint):
            # Resume from the latest checkpoint
            print(
                f"Resuming training from the latest checkpoint: {last_checkpoint}"
            )
            train_model(
                DATA_PATH,
                CHECKPOINTS_PATH,
                resume_from_checkpoint=last_checkpoint,
            )
            print("Training resumed successfully")
        else:
            print("Starting training from scratch")
            train_model(DATA_PATH, CHECKPOINTS_PATH)
    except Exception as e:
        print(f"Training interrupted due to: {str(e)}")
        return None

    return


# ## Run the model
#
# We define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps)
# to run the training.
# If the function times out, or if the job is [preempted](/docs/guide/preemption#preemption), the loop will catch the exception and attempt to resume training from the last checkpoint.

# You can run this locally with `modal run 06_gpu_and_ml.long-training.long-training --detach`
# This runs the code in detached mode, allowing it to continue running even if you close your terminal or computer. This is important since training jobs can be long.


@app.local_entrypoint()
def main():
    while True:
        try:
            print("Starting new training run")
            train.remote()

            print("Finished training")
            break  # Exit the loop if training completes successfully
        except KeyboardInterrupt:
            print("Job was preempted")
            print("Will attempt to resume in the next iteration.")
            continue
        except modal.exception.FunctionTimeoutError:
            print("Function timed out")
            print("Will attempt to resume in the next iteration.")
            continue
        except Exception as e:
            print(f"Error: {str(e)}")
            break
