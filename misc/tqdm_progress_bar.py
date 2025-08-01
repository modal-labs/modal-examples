# # Show a progress bar with tqdm on Modal

# This example shows how you can show a progress bar with [tqdm](https://github.com/tqdm/tqdm) on Modal.

import time

import modal

app = modal.App(
    "example-tqdm",
    image=modal.Image.debian_slim().pip_install("tqdm"),
)


@app.function()
def f():
    from tqdm import tqdm

    for i in tqdm(range(100)):
        time.sleep(0.1)


if __name__ == "__main__":
    with app.run():
        f.remote()
