## Fine-tuning OpenAI's whisper model for improved automatic Hindi speech recognition

The following configuration will finetune the `whisper-small` model for almost 3 hrs,
acheiving a word error rate (WER) of about 55-60. Increasing the number of training
epochs should improve performance, decreasing WER.

You can benchmark this example's performance using Huggingface's [**autoevaluate leaderboard**]https://huggingface.co/spaces/autoevaluate/leaderboards?dataset=mozilla-foundation%2Fcommon_voice_11_0&only_verified=0&task=automatic-speech-recognition&config=hi&split=test&metric=wer).

```bash
python3 -m train \
	--model_name_or_path="openai/whisper-small" \
	--dataset_config_name="clean" \
	--train_split_name="train.100" \
	--eval_split_name="validation" \
	--preprocessing_num_workers="16" \
	--length_column_name="input_length" \
	--num_train_epochs="5" \
	--freeze_feature_encoder=False \
	--per_device_train_batch_size="8" \
	--per_device_eval_batch_size="8" \
	--gradient_accumulation_steps="8" \
	--learning_rate="3e-4" \
	--warmup_steps="400" \
	--evaluation_strategy="steps" \
	--text_column_name="sentence" \
	--save_steps="400" \
	--eval_steps="400" \
	--logging_steps="10" \
	--save_total_limit="3" \
	--freeze_feature_encoder=False \
	--gradient_checkpointing \
	--fp16 \
	--group_by_length \
	--predict_with_generate \
	--generation_max_length="40" \
	--generation_num_beams="1" \
	--do_train --do_eval \
	--do_lower_case
```

### Testing

Use `python3 -m train.end_to_end_check` to do a full train → serialize → save → load → predict
run in less than 5 minutes, checking that the finetuning program is functional.
