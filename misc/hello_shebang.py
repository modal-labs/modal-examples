#!/usr/bin/env python
# # Syntax for making modal scripts executable

# This example shows how you can add a shebang to a script that is meant to be invoked with `modal run`.

import sys

import modal

app = modal.App("example-hello-world")


@app.function()
def f(i):
    if i % 2 == 0:
        print("hello", i)
    else:
        print("world", i, file=sys.stderr)

    return i * i


@app.local_entrypoint()
def main():
    # run the function locally
    print(f.local(1000))

    # run the function remotely on Modal
    print(f.remote(1002))

    # run the function in parallel and remotely on Modal
    total = 0
    for ret in f.map(range(200)):
        total += ret

    print(total)


if __name__ == "__main__":
    # Use `modal.enable_output()` to print the Sandbox's image build logs to the console, just like `modal run` does.
    # Use `app.run()` to substitute the `modal run` CLI invocation.
    with modal.enable_output(), app.run():
        main()
