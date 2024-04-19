import modal

app = modal.App("example-generators")  # Note: prior to April 2024, "app" was called "stub"


@app.function()
def f(i):
    for j in range(i):
        yield j


@app.local_entrypoint()
def main():
    for r in f.remote_gen(10):
        print(r)
