# # Hello, world!
#
# This tutorial demonstrates some core features of Modal:
#
# * You can run functions just as easily locally on your infrastructure as you can remotely on Modal's cloud infrastructure.
# * Logs to `stdout` and `stderr` show up immediately, even for remote functions.
# * Running functions in parallel on Modal is simple and fast.
#
# ## Importing Modal and setting up
#
# We start by importing `modal` and creating a `Stub`.
# We build up from our `Stub` to define our application.

import sys

import modal

stub = modal.Stub("example-hello-world")

# ## Writing some code
#
# Modal takes code and runs it in the cloud.
#
# So first we've got to write some code.
#
# Let's do something simple and silly:
# logging `"hello"` to standard out if the input is even
# or `"world"` to standard error if it's not,
# then returning the input times itself.


@stub.function()
def f(i):
    if i % 2 == 0:
        print("hello", i)
    else:
        print("world", i, file=sys.stderr)

    return i * i


# ## Running our code locally, remotely, and in parallel
#
# Now let's see three different ways we can call that function:
#
# 1. As a regular `local` call on your computer, with `f.local`
#
# 2. As a `remote` call that runs in the cloud, with `f.remote`
#
# 3. By `map`ping many copies of `f` in the cloud over many inputs, with `f.map`
#
# We call `f` in each of these ways inside a `main` function below.


@stub.local_entrypoint()
def main():
    # call the function locally
    print(f.local(1000))

    # call the function remotely
    print(f.remote(1000))

    # run the function in parallel and remotely
    total = 0
    for ret in f.map(range(20)):
        total += ret

    print(total)


# Enter `modal run hello_world.py` in a shell and you'll see
# a Modal app start up. Modal runs functions in _containers_,
# so you'll see some container initialization info.
# Then you'll see the `print`ed logs of
# the `main` function and, mixed in with them, all the logs of `f` as it is run
# locally, then remotely, and then remotely and in parallel across many containers.
#
# That's all triggered by adding the `@stub.local_entrypoint()` decorator on `main`,
# which defines it as the function we start from locally when we invoke `modal run`.
#
# ## But why does this matter?
#
# Try doing one of these things next to start seeing the power of Modal!
#
# ### You can change the code and run it again
#
# For instance, change the `print` statement in the function `f`
# to print `"spam"` and `"eggs"` instead and run the app again.
# You'll see that that your new code is run with no extra work from you --
# and it should even run faster!
#
# Modal's goal is to make running code in the cloud feel like you're
# running code locally. That means no waiting for long image builds when you've just moved a comma,
# no fiddling with container image pushes, and no context-switching to a web UI to inspect logs.
#
# ### You can map over more data
#
# Change the `map` range from `20` to some large number, like `1170`. You'll see
# Modal create and run even more containers in parallel this time.
#
# The function `f` is obviously silly and doesn't do much, but in its place
# imagine something that matters to you, like:
#
# * Running [language model inference](/docs/examples/vllm_mixtral) or [fine-tuning](/docs/examples/slack-finetune)
# * Manipulating [audio](/docs/examples/discord-musicgen) or [images](stable_diffusion_xl_turbo)
# * [Collecting financial data](/docs/examples/fetch_stock_prices) to backtest a trading algorithm.
#
# Modal lets you parallelize that operation effortlessly by running hundreds or
# thousands of containers in the cloud.
