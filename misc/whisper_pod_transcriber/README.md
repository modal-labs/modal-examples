# Modal Whisper Podcast Transcription

This is an example of a full-stack Modal application consisting of 3 components:

1. React + Vite SPA
2. FastAPI server
3. Modal async job queue

The entire application is hosted serverlessly on Modal.

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

Once you have `vite build` running, in a separate shell run `python -m whisper_pod_transcriber.main serve` (from the directory above this one) to start [serving](https://modal.com/docs/reference/modal.Stub#serve) your app on Modal. Pressing `Ctrl+C` will stop your app.

### Deploy to Modal

Once your happy with your changes, run `modal app deploy whisper_pod_transcriber.main serve` (from the directory above this one) to deploy your app to Modal.
