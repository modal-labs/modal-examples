# ---
# integration-test: false
# lambda-test: false
# ---

import modal

stub = modal.Stub(image=modal.DebianSlim().apt_install(["vim"]))

if __name__ == "__main__":
    stub.interactive_shell("/bin/bash")
