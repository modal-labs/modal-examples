import modal

RIFFUSION_PATH = "/root/riffusion"

image = (
    modal.Image.debian_slim()
    .apt_install(["git"])
    .run_commands(
        [
            f"git clone https://github.com/hmartiro/riffusion-inference {RIFFUSION_PATH}",
            f"pip install -r {RIFFUSION_PATH}/requirements.txt",
        ]
    )
)

stub = modal.Stub("example-riffusion", image=image)


@stub.wsgi()
def server():
    import sys

    sys.path.insert(0, RIFFUSION_PATH)
    from riffusion.server import app

    return app


if __name__ == "__main__":
    stub.serve()
