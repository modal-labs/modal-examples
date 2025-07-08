# Modal Podcast Transcriber

This is a complete application that uses [NVIDIA Parakeet ASR](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#parakeet) to transcribe podcasts. Modal spins up multiple containers for a single transcription run, so hours of audio can be transcribed on-demand in a few minutes.

You can find our deployment of the app [here](https://modal-labs-examples--parakeet-pod-transcriber-fastapi-app.modal.run/).

## Architecture

The entire application is hosted serverlessly on Modal and consists of 3 components:

1. React + Vite SPA ([`app/frontend/`](./app/frontend/))
2. FastAPI server ([`app/api.py`](./app/api.py))
3. Modal async job queue ([`app/main.py`](./app/main.py))

## Developing locally

### Requirements

- `npm`
- `modal` installed in your current Python virtual environment

### Podchaser Secret

To run this on your own Modal account, you'll need to [create a Podchaser account and create an API key](https://api-docs.podchaser.com/docs/guides/guide-first-podchaser-query/#getting-your-access-token).

Then, create a [Modal Secret](https://modal.com/secrets/) with the following keys:

- `PODCHASER_CLIENT_SECRET`
- `PODCHASER_CLIENT_ID`

You can find both on [their API page](https://www.podchaser.com/profile/settings/api).

### Vite build

`cd` into the `app/frontend` directory, and run:

- `npm install`
- `npx vite build --watch`

The last command will start a watcher process that will rebuild your static frontend files whenever you make changes to the frontend code.

### Serve on Modal

Once you have `vite build` running, in a separate shell run this to start an ephemeral app on Modal:

```shell
modal serve -m app.main
```

Pressing `Ctrl+C` will stop your app.

### Deploy to Modal

Once your happy with your changes, run `modal deploy -m app.main` to deploy your app to Modal.
