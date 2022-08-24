import modal

from .multiple_modules_2 import square

stub = modal.Stub()


@stub.function
def cube(x):
    return square(x) * x


if __name__ == "__main__":
    with stub.run():
        assert cube(42) == 74088
