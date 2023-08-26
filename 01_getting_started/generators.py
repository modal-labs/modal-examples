import modal

stub = modal.Stub("example-generators")


@stub.function()
def f(i):
    for j in range(i):
        yield j


@stub.local_entrypoint()
def main():
    for r in f.remote_gen(10):
        print(r)

    for r in f.map(range(5)):
        print(r)
