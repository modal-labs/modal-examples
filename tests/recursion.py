import modal

stub = modal.Stub()


@stub.function
def fib(n):
    if n < 2:
        return n
    else:
        return fib(n - 2) + fib(n - 1)


if __name__ == "__main__":
    with stub.run():
        assert fib(3) == 2
