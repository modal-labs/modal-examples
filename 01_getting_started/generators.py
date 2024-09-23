# # Run a generator function on Modal

# This example shows how you can run a generator function on Modal. We define a
# function that `yields` values and then call it with the [`remote_gen`](https://modal.com/docs/reference/modal.Function#remote_gen) method. The
# `remote_gen` method returns a generator object that can be used to iterate over
# the values produced by the function.

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
