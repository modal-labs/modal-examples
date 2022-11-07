# ---
# lambda-test: false
# ---

import modal

stub = modal.Stub("example-shell", image=modal.Image.debian_slim().apt_install(["vim"]))

if __name__ == "__main__":
    stub.interactive_shell("/bin/bash")
