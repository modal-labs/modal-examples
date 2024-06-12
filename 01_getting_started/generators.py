import modal

app = modal.App("example-generators")


@app.function()
def f(i):
    for j in range(i):
        yield j


@app.local_entrypoint()
def main():
    for r in f.remote_gen(10):
        print(r)
