# # Hello, world!
#
# This is a trivial example of a Modal function, but it illustrates a few features:
#
# * You can print things to stdout and stderr.
# * You can return data.
# * You can map over a function.
#
# ## Import Modal and define the app
#
# Let's start with the top level imports.
# You need to import Modal and define the app.
# A stub is an object that defines everything that will be run.

import sys

import modal

stub = modal.Stub("example-hello-world")

# ## Defining a function
#
# Here we define a Modal function using the `modal.function` decorator.
# The body of the function will automatically be run remotely.
# This particular function is pretty silly: it just prints "hello"
# and "world" alternatingly to standard out and standard error.


@stub.function()
def f(i):
    if i % 2 == 0:
        print("hello", i)
    else:
        print("world", i, file=sys.stderr)

    return i * i


# ## Running it
#
# Finally, let's actually invoke it.
# We put this invocation code inside a `@stub.local_entrypoint()`.
# This is because this module will be imported in the cloud, and we don't want
# this code to be executed a second time in the cloud.
#
# Run `modal run hello_world.py` and the `@stub.local_entrypoint()` decorator will handle
# starting the Modal app and then executing the wrapped function body.
#
# Inside the `main()` function body, we are calling the function `f` in three ways:
#
# 1  As a simple local call, `f(1000)`
# 2. As a simple *remote* call `f.call(1000)`
# 3. By mapping over the integers `0..19`


@stub.local_entrypoint()
def main():
    # Call the function locally.
    print(f(1000))

    # Call the function remotely.
    print(f.call(1000))

    # Parallel map.
    total = 0
    for ret in f.map(range(20)):
        total += ret

    print(total)


# ## What happens?
#
# When you do `.call` on function `f`, Modal will execute `f` **in the cloud,**
# not locally on your computer. It will take the code, put it inside a
# container, run it, and stream all the output back to your local
# computer.
#
# Try doing one of these things next.
#
# ### Change the code and run again
#
# For instance, change the `print` statement in the function `f`.
# You can see that the latest code is always run.
#
# Modal's goal is to make running code in the cloud feel like you're
# running code locally. You don't need to run any commands to rebuild,
# push containers, or go to a web UI to download logs.
#
# ### Map over a larger dataset
#
# Change the map range from 20 to some large number. You can see that
# Modal will create and run more containers in parallel.
#
# The function `f` is obviously silly and doesn't do much, but you could
# imagine something more significant, like:
#
# * Training a machine learning model
# * Transcoding media
# * Backtesting a trading algorithm.
#
# Modal lets you parallelize that operation trivially by running hundreds or
# thousands of containers in the cloud.
