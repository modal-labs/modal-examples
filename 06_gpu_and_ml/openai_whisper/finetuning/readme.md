## Fine-tuning OpenAI's whisper model for improved automatic Hindi speech recognition

The following configuration will finetune the `whisper-small` model for almost 3 hrs,
acheiving a word error rate (WER) of about 55-60. Increasing the number of training
epochs should improve performance, decreasing WER.

You can benchmark this example's performance using Huggingface's [**autoevaluate leaderboard**]https://huggingface.co/spaces/autoevaluate/leaderboards?dataset=mozilla-foundation%2Fcommon_voice_11_0&only_verified=0&task=automatic-speech-recognition&config=hi&split=test&metric=wer).

```bash
modal run -m train.train --num_train_epochs=10
```

### Testing

Use `modal run -m train.end_to_end_check` to do a full train → serialize → save → load → predict
run in less than 5 minutes, checking that the finetuning program is functional.
