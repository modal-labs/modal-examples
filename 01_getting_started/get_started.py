import modal

stub = modal.Stub()


@stub.function
def square(x):
    print("This code is running on a remote worker!")
    return x**2


if __name__ == "__main__":
    with stub.run():
        print("the square is", square(42))
