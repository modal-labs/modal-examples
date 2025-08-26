import modal

app = modal.App("docker-file-copy")

image = modal.Image.from_dockerfile("Dockerfile", context_dir=".")


@app.function(image=image)
def hello():
    with open("/tmp/00_hello.sh") as f:
        print(f.read())
