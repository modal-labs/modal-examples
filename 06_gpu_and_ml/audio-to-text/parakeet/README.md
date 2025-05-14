# Parakeet Websocket Example

## Installation

1. Create a new venv for client `python -m venv .venv && source .venv/bin/activate`.
   `pip install -r requirements.txt` for client requirements.
2. Run the websocket backend: `modal deploy api.py`
3. Retrieve the URL - should be something like `wss://{workspace_name}-{environment}--parakeet-websocket-parakeet-web.modal.run`. Make sure to use the `wss` prefix, not `https`!
4. Run the client `python client.py --=url={your URL}`
5. You should see something like this:

```
☀️ Waking up model, this may take a few seconds on cold start...

Recording and streaming... Press Ctrl+C to stop.
📝 Transcription: Hi, how's it going?.
📝 Transcription: Doing well. Great day to be having fun with transcription models
📝 Transcription:
📝 Transcription: Good.
^C
🛑 Stopped by user.
```
