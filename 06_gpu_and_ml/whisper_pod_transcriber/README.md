# Modal Podcast Transcriber

This is a complete application that uses [OpenAI Whisper](https://github.com/openai/whisper) to transcribe podcasts. Modal spins up 100-300 containers for a single transcription run, so hours of audio can be transcribed on-demand in a few minutes.

You can find the app here: https://modal-labs--whisper-pod-transcriber-fastapi-app.modal.run/

## Architecture

The entire application is hosted serverlessly on Modal and consists of 3 components:

1. React + Vite SPA (`whisper_frontend/`)
2. FastAPI server (`./api.py`)
3. Modal async job queue (`./main.py`)

## Developing locally

### Requirements

- npm
- `modal` installed in your current Python virtual environment

### Podchaser Secret

To run this on your own Modal account, you'll need to [create a Podchaser account and create an API key](https://api-docs.podchaser.com/docs/guides/guide-first-podchaser-query/#getting-your-access-token).

Then, create a [Modal Secret](https://modal.com/secrets/) with the following keys:

- `PODCHASER_CLIENT_SECRET`
- `PODCHASER_CLIENT_ID`

You can find both on [their API page](https://www.podchaser.com/profile/settings/api).

### Vite build

`cd` into the `whisper_frontend` directory, and run:

- `npm install`
- `npx vite build --watch`

The last command will start a watcher process that will rebuild your static frontend files whenever you make changes to the frontend code.

### Serve on Modal

Once you have `vite build` running, in a separate shell run (from the directory above this one)
```shell
modal serve whisper_pod_transcriber.main
```
to start an ephemeral app on Modal. Pressing `Ctrl+C` will stop your app.


### Deploy to Modal

Once your happy with your changes, run `modal deploy whisper_pod_transcriber.main` (from the directory above this one) to deploy your app to Modal.
