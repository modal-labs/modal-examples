# Fine-tuning the OpenAI Whisper model on Modal for improved
# transcription performance on the Hindi language.
#
# Based on the work done in https://huggingface.co/blog/fine-tune-whisper.

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import modal

from .config import DataTrainingArguments, ModelArguments, app_config
from .logs import get_logger, setup_logging

persistent_volume = modal.Volume.from_name(
    "example-whisper-fine-tune-vol",
    create_if_missing=True,
)

image = modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements(
    "requirements.txt"
)
app = modal.App(
    name="example-whisper-fine-tune",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)

logger = get_logger(__name__)


@app.function(
    gpu="A10G",
    volumes={app_config.model_dir: persistent_volume},
    # 12hrs
    timeout=12 * 60 * 60,
    # For occasional connection error to 'cdn-lfs.huggingface.co'
    retries=1,
)
def train(
    num_train_epochs: int = 5,
    warmup_steps: int = 400,
    max_steps: int = -1,
    overwrite_output_dir: bool = False,
):
    import datasets
    import evaluate
    import torch
    from datasets import DatasetDict, load_dataset
    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        AutoTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint, is_main_process

    model_args = ModelArguments(
        model_name_or_path="openai/whisper-small",
        freeze_feature_encoder=False,
    )

    run_id = app.app_id
    output_dir = Path(app_config.model_dir, run_id).as_posix()

    data_args = DataTrainingArguments(
        dataset_config_name="clean",
        train_split_name="train.100",
        eval_split_name="validation",
        text_column_name="sentence",
        preprocessing_num_workers=16,
        max_train_samples=5,
        max_eval_samples=5,
        do_lower_case=True,
    )

    training_args = Seq2SeqTrainingArguments(
        length_column_name="input_length",
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=3e-4,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        evaluation_strategy="steps",
        save_total_limit=3,
        gradient_checkpointing=True,
        fp16=True,
        group_by_length=True,
        predict_with_generate=True,
        generation_max_length=40,
        generation_num_beams=1,
        do_train=True,
        do_eval=True,
    )

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            processor ([`WhisperProcessor`])
                The processor used for processing the data.
            decoder_start_token_id (`int`)
                The begin-of-sentence of the decoder.
            forward_attention_mask (`bool`)
                Whether to return attention_mask.
        """

        processor: Any
        decoder_start_token_id: int
        forward_attention_mask: bool

        def __call__(
            self, features: list[dict[str, Union[list[int], torch.Tensor]]]
        ) -> dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need
            # different padding methods
            model_input_name = self.processor.model_input_names[0]
            input_features = [
                {model_input_name: feature[model_input_name]} for feature in features
            ]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )

            if self.forward_attention_mask:
                batch["attention_mask"] = torch.LongTensor(
                    [feature["attention_mask"] for feature in features]
                )

            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels

            return batch

    logger.info("Starting training run")
    logger.info(f"Finetuned model will be persisted to '{training_args.output_dir}'")
    setup_logging(
        logger=logger,
        log_level=training_args.get_process_log_level(),
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    logger.info(
        "3. Detecting last checkpoint and eventually continue from last checkpoint"
    )
    last_checkpoint = None
    if (
        Path(training_args.output_dir).exists()
        and training_args.do_train
        and not overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            print(os.listdir(training_args.output_dir))
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    logger.info("4. Load datasets")
    raw_datasets = DatasetDict()
    raw_datasets["train"] = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "hi",
        split="train+validation",
        trust_remote_code=True,
    )
    raw_datasets["eval"] = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "hi",
        split="test",
    )

    # Most ASR datasets only provide input audio samples (audio) and
    # the corresponding transcribed text (sentence).
    # Common Voice contains additional metadata information,
    # such as accent and locale, which we can disregard for ASR.
    # Keeping the training function as general as possible,
    # we only consider the input audio and transcribed text for fine-tuning,
    # discarding the additional metadata information:
    raw_datasets = raw_datasets.remove_columns(
        [
            "accent",
            "age",
            "client_id",
            "down_votes",
            "gender",
            "locale",
            "path",
            "segment",
            "up_votes",
        ]
    )

    logger.info("5. Load pretrained model, tokenizer, and feature extractor")
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=os.environ["HF_TOKEN"],
    )

    config.update(
        {
            "forced_decoder_ids": model_args.forced_decoder_ids,
            "suppress_tokens": model_args.suppress_tokens,
        }
    )
    # SpecAugment for whisper models
    config.update({"apply_spec_augment": model_args.apply_spec_augment})

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        (
            model_args.feature_extractor_name
            if model_args.feature_extractor_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if model_args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    if data_args.language is not None:
        # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
        tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task)

    logger.info("6. Resample speech dataset if necessary")
    dataset_sampling_rate = (
        next(iter(raw_datasets.values()))
        .features[data_args.audio_column_name]
        .sampling_rate
    )
    if dataset_sampling_rate != feature_extractor.sampling_rate:
        logger.info("Resampling necessary")
        raw_datasets = raw_datasets.cast_column(
            data_args.audio_column_name,
            datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
        )

    logger.info("7. Preprocessing the datasets.")
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = (
        data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    )
    min_input_length = (
        data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    )
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    text_column_name = data_args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    do_lower_case = data_args.do_lower_case
    # if SpecAugment is used for whisper models, return attention_mask to guide the mask along time axis
    forward_attention_mask = (
        getattr(config, "model_type", None) == "whisper"
        and getattr(config, "apply_spec_augment", False)
        and getattr(config, "mask_time_prob", 0) > 0
    )

    if data_args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(
            range(data_args.max_train_samples)
        )

    if data_args.max_eval_samples is not None:
        raw_datasets["eval"] = raw_datasets["eval"].select(
            range(data_args.max_eval_samples)
        )

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(
            sample["array"],
            sampling_rate=sample["sampling_rate"],
            return_attention_mask=forward_attention_mask,
        )
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(sample["array"])
        if forward_attention_mask:
            batch["attention_mask"] = inputs.get("attention_mask")[0]

        # process targets
        input_str = (
            batch[text_column_name].lower()
            if do_lower_case
            else batch[text_column_name]
        )
        batch["labels"] = tokenizer(input_str).input_ids
        return batch

    with training_args.main_process_first(desc="dataset map pre-processing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=data_args.preprocessing_num_workers,
            desc="preprocess train dataset",
        )

    # filter data that is shorter than min_input_length or longer than
    # max_input_length
    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    vectorized_datasets = vectorized_datasets.filter(
        is_audio_in_length_range,
        num_proc=num_workers,
        input_columns=["input_length"],
    )

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return

    logger.info("8. Loading WER Metric")
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

        wer = metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    logger.info("9. Create a single speech processor")
    # make sure all processes wait until data is saved
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_rank):
            logger.info("saving feature extractor, tokenizer and config")
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    processor = AutoProcessor.from_pretrained(training_args.output_dir)

    logger.info("10. Constructing data collator")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
    )

    logger.info("11. Initializing Trainer class")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=(
            vectorized_datasets["train"] if training_args.do_train else None
        ),
        eval_dataset=(vectorized_datasets["eval"] if training_args.do_eval else None),
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=(
            compute_metrics if training_args.predict_with_generate else None
        ),
    )

    logger.info("12. Running training")
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            logger.info("Restoring from previous training checkpoint")
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        logger.info("Saving model")
        trainer.save_model()  # Saves the feature extractor too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(
            max_train_samples, len(vectorized_datasets["train"])
        )
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        persistent_volume.commit()

    logger.info("13. Running evaluation")
    results = {}  # type: ignore
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(vectorized_datasets["eval"])
        )
        metrics["eval_samples"] = min(
            max_eval_samples, len(vectorized_datasets["eval"])
        )

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("14. Write training stats")
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "automatic-speech-recognition",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = (
                f"{data_args.dataset_name} {data_args.dataset_config_name}"
            )
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    logger.info("Training run complete!")
    return results
