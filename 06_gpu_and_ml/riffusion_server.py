import modal

COMMIT_HASH = "cae8a3892e03810444c8574179e7cd065bbcdb26"
RIFFUSION_PKG_PATH = "/root/riffusion"
MODEL_CACHE_PATH = "/cache"

inference_image = (
    modal.Image.debian_slim()
    .apt_install("git", "ffmpeg")
    .run_commands(
        f"git clone https://github.com/riffusion/riffusion {RIFFUSION_PKG_PATH}",
        f"cd {RIFFUSION_PKG_PATH} && git checkout {COMMIT_HASH}",
        f"pip install -r {RIFFUSION_PKG_PATH}/requirements.txt",
    )
)

stub = modal.Stub("example-riffusion")

volume = modal.SharedVolume().persist("riffusion-model-vol")


@stub.wsgi(image=inference_image, gpu="A10g", shared_volumes={MODEL_CACHE_PATH: volume})
def server():
    import os
    import sys

    # HACK since riffusion is not a package, but we want to use it like one.
    sys.path.insert(0, RIFFUSION_PKG_PATH)

    os.environ["HF_HOME"] = MODEL_CACHE_PATH

    import riffusion
    from riffusion.server import app, load_model

    # HACK set the global variable where the server expects to find the model
    riffusion.server.MODEL = load_model(checkpoint="riffusion/riffusion-model-v1", traced_unet=True)

    return app


# web_image = (
#     modal.Image.debian_slim()
#     .apt_install(["git", "curl"])
#     # Install npm
#     .run_commands(
#         [
#             "curl https://raw.githubusercontent.com/creationix/nvm/master/install.sh | bash",
#             ". $HOME/.nvm/nvm.sh && nvm install 16.14.2",
#         ]
#     )
#     .run_commands(
#         [
#             f"git clone https://github.com/hmartiro/riffusion-app.git ~/riffusion-app",
#             ". $HOME/.nvm/nvm.sh && cd ~/riffusion-app && npm install && next build && next export",
#         ]
#     )
# )

# @stub.wsgi(image=web_image)
# def web_server():
#     from riffusion.server import app

#     return app


if __name__ == "__main__":
    stub.serve()
