import time

import modal

stub = modal.Stub(image=modal.Image.debian_slim().pip_install(["tqdm"]))


@stub.function
def f():
    from tqdm import tqdm

    for i in tqdm(range(100)):
        time.sleep(0.1)


if __name__ == "__main__":
    with stub.run():
        f()
