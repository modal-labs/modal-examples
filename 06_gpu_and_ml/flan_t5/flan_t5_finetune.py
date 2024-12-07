# # Finetuning Flan-T5

# Example by [@anishpdalal](https://github.com/anishpdalal)

# [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) is a highly versatile model that's been instruction-tuned to
# perform well on a variety of text-based tasks such as question answering and summarization. There are smaller model variants available which makes
# Flan-T5 a great base model to use for finetuning on a specific instruction dataset with just a single GPU. In this example, we'll
# finetune Flan-T5 on the [Extreme Sum ("XSum")](https://huggingface.co/datasets/xsum) dataset to summarize news articles.

# ## Defining dependencies

# The example uses the `dataset` package from HuggingFace to load the xsum dataset. It also uses the `transformers`
# and `accelerate` packages with a PyTorch backend to finetune and serve the model. Finally, we also
# install `tensorboard` and serve it via a web app. All packages are installed into a Debian Slim base image
# using the `pip_install` function.

from pathlib import Path

import modal

VOL_MOUNT_PATH = Path("/vol")

# Other Flan-T5 models can be found [here](https://huggingface.co/docs/transformers/model_doc/flan-t5)
BASE_MODEL = "google/flan-t5-base"

image = modal.Image.debian_slim().pip_install(
    "accelerate",
    "transformers",
    "torch",
    "datasets",
    "tensorboard",
)

app = modal.App(name="example-news-summarizer", image=image)
output_vol = modal.Volume.from_name("finetune-volume", create_if_missing=True)

# ### Handling preemption

# As this finetuning job is long-running it's possible that it experiences a preemption.
# The training code is robust to pre-emption events by periodically saving checkpoints and restoring
# from checkpoint on restart. But it's also helpful to observe in logs when a preemption restart has occurred,
# so we track restarts with a `modal.Dict`.

# See the [guide on preemptions](/docs/guide/preemption#preemption) for more details on preemption handling.

restart_tracker_dict = modal.Dict.from_name(
    "finetune-restart-tracker", create_if_missing=True
)


def track_restarts(restart_tracker: modal.Dict) -> int:
    if not restart_tracker.contains("count"):
        preemption_count = 0
        print(f"Starting first time. {preemption_count=}")
        restart_tracker["count"] = preemption_count
    else:
        preemption_count = restart_tracker.get("count") + 1
        print(f"Restarting after pre-emption. {preemption_count=}")
        restart_tracker["count"] = preemption_count
    return preemption_count


# ## Finetuning Flan-T5 on XSum dataset

# Each row in the dataset has a `document` (input news article) and `summary` column.


@app.function(
    gpu="A10g",
    timeout=7200,
    volumes={VOL_MOUNT_PATH: output_vol},
)
def finetune(num_train_epochs: int = 1, size_percentage: int = 10):
    from datasets import load_dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    restarts = track_restarts(restart_tracker_dict)

    # Use size percentage to retrieve subset of the dataset to iterate faster
    if size_percentage:
        xsum_train = load_dataset("xsum", split=f"train[:{size_percentage}%]")
        xsum_test = load_dataset("xsum", split=f"test[:{size_percentage}%]")

    # Load the whole dataset
    else:
        xsum = load_dataset("xsum")
        xsum_train = xsum["train"]
        xsum_test = xsum["test"]

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    # Replace all padding tokens with a large negative number so that the loss function ignores them in
    # its calculation
    padding_token_id = -100

    batch_size = 8

    def preprocess(batch):
        # prepend summarize: prefix to document to convert the example to a summarization instruction
        inputs = ["summarize: " + doc for doc in batch["document"]]

        model_inputs = tokenizer(
            inputs, max_length=512, truncation=True, padding="max_length"
        )

        labels = tokenizer(
            text_target=batch["summary"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )

        labels["input_ids"] = [
            [
                l if l != tokenizer.pad_token_id else padding_token_id
                for l in label
            ]
            for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_xsum_train = xsum_train.map(
        preprocess, batched=True, remove_columns=["document", "summary", "id"]
    )

    tokenized_xsum_test = xsum_test.map(
        preprocess, batched=True, remove_columns=["document", "summary", "id"]
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=padding_token_id,
        pad_to_multiple_of=batch_size,
    )

    training_args = Seq2SeqTrainingArguments(
        # Save checkpoints to the mounted volume
        output_dir=str(VOL_MOUNT_PATH / "model"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        learning_rate=3e-5,
        num_train_epochs=num_train_epochs,
        logging_strategy="steps",
        logging_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_xsum_train,
        eval_dataset=tokenized_xsum_test,
    )

    try:
        resume = restarts > 0
        if resume:
            print("resuming from checkpoint")
        trainer.train(resume_from_checkpoint=resume)
    except KeyboardInterrupt:  # handle possible preemption
        print("received interrupt; saving state and model")
        trainer.save_state()
        trainer.save_model()
        raise

    # Save the trained model and tokenizer to the mounted volume
    model.save_pretrained(str(VOL_MOUNT_PATH / "model"))
    tokenizer.save_pretrained(str(VOL_MOUNT_PATH / "tokenizer"))
    output_vol.commit()
    print("✅ done")


# ## Monitoring Finetuning with Tensorboard

# Tensorboard is an application for visualizing training loss. In this example we
# serve it as a Modal WSGI app.


@app.function(volumes={VOL_MOUNT_PATH: output_vol})
@modal.wsgi_app()
def monitor():
    import tensorboard

    board = tensorboard.program.TensorBoard()
    board.configure(logdir=f"{VOL_MOUNT_PATH}/logs")
    (data_provider, deprecated_multiplexer) = board._make_data_provider()
    wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
        board.flags,
        board.plugin_loaders,
        data_provider,
        board.assets_zip_provider,
        deprecated_multiplexer,
    )
    return wsgi_app


# ## Model Inference
#


@app.cls(volumes={VOL_MOUNT_PATH: output_vol})
class Summarizer:
    @modal.enter()
    def load_model(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

        # Load saved tokenizer and finetuned from training run
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, cache_dir=VOL_MOUNT_PATH / "tokenizer/"
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL, cache_dir=VOL_MOUNT_PATH / "model/"
        )

        self.summarizer = pipeline(
            "summarization", tokenizer=tokenizer, model=model
        )

    @modal.method()
    def generate(self, input: str) -> str:
        return self.summarizer(input)[0]["summary_text"]


@app.local_entrypoint()
def main():
    input = """
    The 14-time major champion, playing in his first full PGA Tour event for almost 18 months,
    carded a level-par second round of 72, but missed the cut by four shots after his first-round 76.
    World number one Jason Day and US Open champion Dustin Johnson also missed the cut at Torrey Pines in San Diego.
    Overnight leader Rose carded a one-under 71 to put him on eight under. Canada's
    Adam Hadwin and USA's Brandt Snedeker are tied in second on seven under, while US PGA champion
    Jimmy Walker missed the cut as he finished on three over. Woods is playing in just his
    second tournament since 15 months out with a back injury. "It's frustrating not being
    able to have a chance to win the tournament," said the 41-year-old, who won his last major,
    the US Open, at the same course in 2008. "Overall today was a lot better than yesterday.
    I hit it better, I putted well again. I hit a lot of beautiful putts that didn't go in, but
    I hit it much better today, which was nice." Scotland's Martin Laird and England's Paul Casey
    are both on two under, while Ireland's Shane Lowry is on level par.
    """
    model = Summarizer()
    response = model.generate.remote(input)
    print(response)


# ## Run via the CLI

# Trigger model finetuning using the following command:

# ```bash
# modal run --detach flan_t5_finetune.py::finetune --num-train-epochs=1 --size-percentage=10
# View the tensorboard logs at https://<username>--example-news-summarizer-monitor-dev.modal.run
# ```

# Then, you can invoke inference via the `local_entrypoint` with this command:

# ```bash
# modal run flan_t5_finetune.py
# World number one Tiger Woods missed the cut at the US Open as he failed to qualify for the final round of the event in Los Angeles.
# ```
