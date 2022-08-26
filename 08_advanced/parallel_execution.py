import time
import modal

stub = modal.Stub()



@stub.function
def step1(word):
    time.sleep(7)
    print("step1 done")
    return word

@stub.function
def step2(number):
    time.sleep(3)
    print("step2 done")
    return 10 // number


if __name__ == "__main__":
    with stub.run() as app:
        # .submit() starts running a function and returns a FunctionCall handle immediately
        word_call = step1.submit("foo")
        number_call = step2.submit(5)
        # prints "foofoo" after 7 seconds
        print(word_call.get() * number_call.get())

        # alternatively, use `modal.gather(...)` as a convenience wrapper
        # which also makes sure to raise an error as soon as the first
        # failing function call fails


        # raises ZeroDivisionError after 3 seconds:
        word, number = modal.functions.gather(
            step1.submit("bar"),
            step2.submit(0)
        )
