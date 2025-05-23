# # Hello, world!

# This tutorial demonstrates some core features of Modal:

# * You can run functions on Modal just as easily as you run them locally.
# * Running functions in parallel on Modal is simple and fast.
# * Logs and errors show up immediately, even for functions running on Modal.

# ## Importing Modal and setting up

# We start by importing `modal` and creating a `App`.
# We build up this `App` to [define our application](https://modal.com/docs/guide/apps).

import sys

import modal

app = modal.App("example-hello-world")

# ## Defining a function

# Modal takes code and runs it in the cloud.

# So first we've got to write some code.

# Let's write a simple function that takes in an input,
# prints a log or an error to the console,
# and then returns an output.

# To make this function work with Modal, we just wrap it in a decorator,
# [`@app.function`](https://modal.com/docs/reference/modal.App#function).


@app.function()
def f(i):
    if i % 2 == 0:
        print("hello", i)
    else:
        print("world", i, file=sys.stderr)

    return i * i


# ## Running our function locally, remotely, and in parallel

# Now let's see three different ways we can call that function:

# 1. As a regular call on your `local` machine, with `f.local`

# 2. As a `remote` call that runs in the cloud, with `f.remote`

# 3. By `map`ping many copies of `f` in the cloud over many inputs, with `f.map`

# We call `f` in each of these ways inside the `main` function below.


@app.local_entrypoint()
def main():
    # run the function locally
    print(f.local(1000))

    # run the function remotely on Modal
    print(f.remote(1000))

    # run the function in parallel and remotely on Modal
    total = 0
    for ret in f.map(range(200)):
        total += ret

    print(total)


# Enter `modal run hello_world.py` in a shell, and you'll see a Modal app initialize.
# You'll then see the `print`ed logs of
# the `main` function and, mixed in with them, all the logs of `f` as it is run
# locally, then remotely, and then remotely and in parallel.

# That's all triggered by adding the
# [`@app.local_entrypoint`](https://modal.com/docs/reference/modal.App#local_entrypoint)
# decorator on `main`, which defines it as the function to start from locally when we invoke `modal run`.

# ## What just happened?

# When we called `.remote` on `f`, the function was executed
# _in the cloud_, on Modal's infrastructure, not on the local machine.

# In short, we took the function `f`, put it inside a container,
# sent it the inputs, and streamed back the logs and outputs.

# ## But why does this matter?

# Try one of these things next to start seeing the full power of Modal!

# ### You can change the code and run it again

# For instance, change the `print` statement in the function `f`
# to print `"spam"` and `"eggs"` instead and run the app again.
# You'll see that that your new code is run with no extra work from you --
# and it should even run faster!

# Modal's goal is to make running code in the cloud feel like you're
# running code locally. That means no waiting for long image builds when you've just moved a comma,
# no fiddling with container image pushes, and no context-switching to a web UI to inspect logs.

# ### You can map over more data

# Change the `map` range from `200` to some large number, like `1170`. You'll see
# Modal create and run even more containers in parallel this time.

# And it'll happen lightning fast!

# ### You can run a more interesting function

# The function `f` is a bit silly and doesn't do much, but in its place
# imagine something that matters to you, like:

# * Running [language model inference](https://modal.com/docs/examples/vllm_inference)
# or [fine-tuning](https://modal.com/docs/examples/slack-finetune)
# * Manipulating [audio](https://modal.com/docs/examples/musicgen)
# or [images](https://modal.com/docs/examples/diffusers_lora_finetune)
# * [Embedding huge text datasets](https://modal.com/docs/examples/amazon_embeddings) at lightning fast speeds

# Modal lets you parallelize that operation effortlessly by running hundreds or
# thousands of containers in the cloud.
