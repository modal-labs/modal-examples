import modal

image = modal.Image.debian_slim().pip_install("openai")
stub = modal.Stub(
    name="example-chatgpt-stream",
    image=image,
    secrets=[modal.Secret.from_name("openai-secret")],
)

@stub.function()
def stream_chat(prompt: str):
    import openai
    
    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        stream=True,
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            yield content


@stub.local_entrypoint()
def main():
    demo_prompt = "Generate a list of 20 great names for sentient cheesecakes that teach SQL"
    for part in stream_chat.call(prompt=demo_prompt):
        print(part, end="")
