import time

import modal

app = modal.App(
    "example-tqdm",
    image=modal.Image.debian_slim().pip_install("tqdm"),
)  # Note: prior to April 2024, "app" was called "stub"


@app.function()
def f():
    from tqdm import tqdm

    for i in tqdm(range(100)):
        time.sleep(0.1)


if __name__ == "__main__":
    with app.run():
        f.remote()
