# # Async functions
#
# Modal natively supports async/await syntax using asyncio.

# First, let's import some global stuff.

import sys

import modal

app = modal.App("example-hello-world-async")


# ## Defining a function
#
# Now, let's define a function. The wrapped function can be synchronous or
# asynchronous, but calling it in either context will still work.
# Let's stick to a normal synchronous function


@app.function()
def f(i):
    if i % 2 == 0:
        print("hello", i)
    else:
        print("world", i, file=sys.stderr)

    return i * i


# ## Running the app with asyncio
#
# Let's make the main entrypoint asynchronous. In async contexts, we should
# call the function using `await` or iterate over the map using `async for`.
# Otherwise we would block the event loop while our call is being run.


@app.local_entrypoint()
async def run_async():
    # Call the function using .remote.aio() in order to run it asynchronously
    print(await f.remote.aio(1000))

    # Parallel map.
    total = 0
    # Call .map asynchronously using using f.map.aio(...)
    async for ret in f.map.aio(range(20)):
        total += ret

    print(total)
