# Audio Diarization Benchmarking

This folder contains code that compares different ASR models using
Modal. Models are compared against a sample dataset of WAV files. Available
models:

- [**nvidia/parakeet-tdt-0.6b-v2**](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2): using NeMo as inference engine
- [**whisper-large-v3-turbo**](https://huggingface.co/openai/whisper-large-v3-turbo): using vLLM
- [**whisperx**](https://github.com/m-bain/whisperX): using Python

If you're familiar with all the functions here, feel free to kick them all off with

```
modal run main.py
```

## Data preparation / Pre-requisites

## Create secret

First, create a [Modal Secret](https://modal.com/docs/guide/secrets#secrets) with Huggingface token.
We'll use this secret later to download models from the Huggingface Hub. The Secret needs to be
named `huggingface-token`. This is required for our WhisperX model. You can skip this step if you only
want to benchmark Whisper or Parakeet

## Upload and prepare dataset

We'll use the publicly available [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/) to benchmark our models

Run the following script to upload all data to a new [Modal Volume](https://modal.com/docs/guide/volumes#volumes)
and convert all WAV files to equivalent files but in 16khz and mono. This makes files compatible with Parakeet. The data in this volume can be accessed by all apps.

```shell
modal run download_and_upload_lj_data.py
```

# Inference

You can benchmark just one model by `modal run`ning either of the files prefixed by `benchmark_`, such as

```
modal run benchmark_parakeet.py
```

Modal will scale to add as many GPUs as necessary in order to process your
dataset. Ouputs will be available in a local CSV file named:

```shell
result_parakeet_$TIMESTAMP.csv
```

## Post-Process Results
