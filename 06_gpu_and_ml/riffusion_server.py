import modal

RIFFUSION_PKG_PATH = "/root/riffusion"

inference_image = (
    modal.Image.debian_slim()
    .apt_install(["git"])
    .run_commands(
        [
            f"git clone https://github.com/hmartiro/riffusion-inference {RIFFUSION_PKG_PATH}",
            f"pip install -r {RIFFUSION_PKG_PATH}/requirements.txt",
        ]
    )
)

stub = modal.Stub("example-riffusion")


class Riffusion:
    def __enter__(self):
        print("loading")
        import sys

        # HACK since riffusion is not a package, but we want to use it like one.
        sys.path.insert(0, RIFFUSION_PKG_PATH)
        from riffusion.server import load_model

        load_model()
        print("loaded")

    @stub.wsgi(image=inference_image)
    def server(self):
        from riffusion.server import app

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
