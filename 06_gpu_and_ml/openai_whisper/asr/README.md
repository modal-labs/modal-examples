A blog post with the details of inner workings: https://huggingface.co/blog/asr-diarization

Use with a prebuilt image:
```
docker run --gpus all -p 7860:7860 --env-file .env ghcr.io/plaggy/asrdiarization-server:latest
```
and parametrize via `.env`:
```
ASR_MODEL=
DIARIZATION_MODEL=
ASSISTANT_MODEL=
HF_TOKEN=
```
Or build your own

Once deployed, send your audio with inference parameters like this:
```python
import requests
import json
import aiohttp

# synchronous call
def sync_post():
    files = {"file": open("<path/to/audio>", "rb")}
    data = {"parameters": json.dumps({"batch_size": 1, "assisted": "true"})}
    resp = requests.post("<ENDPOINT_URL>", files=files, data=data)
    print(resp.json())

# asynchronous call
async def async_post():
    data = {
        "file": open("<path/to/audio>", "rb"),
        "parameters": json.dumps({"batch_size": 30})
    }
    async with aiohttp.ClientSession() as session:
        response = await session.post("<ENDPOINT_URL>", data=data)
        print(await response.json())
```
