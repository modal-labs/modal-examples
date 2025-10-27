import modal

app = modal.App("example-inference")

image = (  # start from base image
    modal.Image.debian_slim()
    .uv_pip_install(  # install Python packages
        "transformers[torch]"
    )
    .apt_install("ffmpeg")  # add audio processing tools
)


@app.function(gpu="l40s", image=image)
def transcribe(url: str | None = None):
    from transformers import pipeline

    transcriber = pipeline(model="openai/whisper-tiny.en", device="cuda")
    result = transcriber(url or "https://modal-cdn.com/mlk.flac")
    print(  # I have a dream that one day this nation will rise up live out the true meaning of its creed
        result["text"]
    )
