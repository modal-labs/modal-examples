import modal

stub = modal.Stub("example-generators")


@stub.function
def f(i):
    for j in range(i):
        yield j


if __name__ == "__main__":
    with stub.run():
        for r in f(10):
            print(r)

        for r in f.map(range(5)):
            print(r)
