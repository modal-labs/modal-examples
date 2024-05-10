"""This simple script shows how to interact with an OpenAI-compatible server from a client."""
import modal
from openai import OpenAI


class Colors:
    """ANSI color codes"""

    GREEN = "\033[0;32m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"


client = OpenAI(api_key="super-secret-token")

WORKSPACE = modal.config._profile

client.base_url = (
    f"https://{WORKSPACE}--vllm-openai-compatible-serve.modal.run/v1"
)

print(
    Colors.GREEN,
    Colors.BOLD,
    f"🧠: Looking up available models on server at {client.base_url}. This may trigger a boot!",
    Colors.END,
    sep="",
)
model = client.models.list().data[0]

print(
    Colors.GREEN,
    Colors.BOLD,
    f"🧠: Requesting completion from model {model.id}",
    Colors.END,
    sep="",
)

messages = [
    {
        "role": "system",
        "content": "You are a poetic assistant, skilled in writing satirical doggerel with creative flair.",
    },
    {
        "role": "user",
        "content": "Compose a limerick about baboons and racoons.",
    },
]

for message in messages:
    color = Colors.GRAY
    emoji = "👉"
    if message["role"] == "user":
        color = Colors.GREEN
        emoji = "👤"
    elif message["role"] == "assistant":
        color = Colors.BLUE
        emoji = "🤖"
    print(
        color,
        f"{emoji}: {message['content']}",
        Colors.END,
        sep="",
    )

stream = client.chat.completions.create(
    model=model.id,
    messages=messages,
    stream=True,
)
print(Colors.BLUE, "🤖:", sep="", end="")
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
print(Colors.END)
