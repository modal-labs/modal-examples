# ---
# runtimes: ["runc", "gvisor"]
# ---
#
# # Using the ChatGPT streaming API
#
# This example shows how to stream from the ChatGPT API as the model is generating a completion, instead of
# waiting for the entire completion to finish. This provides a much better user experience, and is what you
# get when playing with ChatGPT on [chat.openai.com](https://chat.openai.com/).
#
# You can try this out from the command line using the `modal` CLI, or serve the application and use the
# included web endpoint.
#
# ## Imports and Modal application configuration
#
# OpenAI's Python client library is the only package dependency we need. We also need an API key.
# The former is specified in the Modal application's `image` definition, and the latter is attached to the app's
# stub as a [`modal.Secret`](/docs/guide/secrets).

from modal import Image, Secret, Stub, web_endpoint

image = Image.debian_slim(python_version="3.11").pip_install("openai==1.8.0")
stub = Stub(
    name="example-chatgpt-stream",
    image=image,
    secrets=[Secret.from_name("openai-secret")],
)

# This is all the code needed to stream answers back from ChatGPT.
# Not much code to worry about!
#
# Because this Python function is decorated with `@stub.function`, it becomes
# callable as a Modal remote function. But `stream_chat` can still be used as a
# regular Python function, which becomes important below.


def stream_chat(prompt: str):
    import openai

    client = openai.OpenAI()
    for chunk in client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        content = chunk.choices[0].delta.content
        print(content)
        if content is not None:
            yield content


# ## Streaming web endpoint
#
# These four lines are all you need to take that function above and serve it
# over HTTP. It is a single function definition, annotated with decorators to make
# it a Modal function [with a web serving capability](/docs/guide/webhooks).
#
# Notice that the `stream_chat` function is passed into the retuned streaming response.
# This works because the function is a generator and is thus compatible with streaming.
#
# We use the standard Python calling convention `stream_chat(...)` and not the
# Modal-specific calling convention `stream_chat.remote(...)`. The latter would still work,
# but it would create a remote function invocation which would unnecessarily involve `stream_chat`
# running in a separate container, sending its results back to the caller over the network.


@stub.function()
@web_endpoint()
def web(prompt: str):
    from fastapi.responses import StreamingResponse

    return StreamingResponse(stream_chat(prompt), media_type="text/event-stream")


# ## Try out the web endpoint
#
# Run this example with `modal serve chatgpt_streaming.py` and you'll see an ephemeral web endpoint
# has started serving. Hit this endpoint with a prompt and watch the ChatGPT response streaming back in
# your browser or terminal window.
#
# We've also already deployed this example and so you can try out our deployed web endpoint:
#
# ```bash
# curl --get \
#   --data-urlencode "prompt=Generate a list of 20 great names for sentient cheesecakes that teach SQL" \
#   https://modal-labs--example-chatgpt-stream-web.modal.run
# ```
#
# ## CLI interface
#
# Doing `modal run chatgpt_streaming.py --prompt="Generate a list of the world's most famous people"` also works, and uses the `local_entrypoint` defined below.

default_prompt = (
    "Generate a list of 20 great names for sentient cheesecakes that teach SQL"
)


@stub.local_entrypoint()
def main(prompt: str = default_prompt):
    for part in stream_chat.remote_gen(prompt=prompt):
        print(part, end="")
