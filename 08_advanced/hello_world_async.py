# # Async functions
#
# Modal natively supports async/await syntax using asyncio.

# First, let's import some global stuff.

import asyncio
import sys

import modal.aio

# ## Using Modal asynchronously
#
# If you want to use Modal asynchronously, you need to import `modal.aio` and use classes prefixed by `Aio`.
# In this case, we just need to define an asynchronous stub:


stub = modal.aio.AioStub()


# ## Defining a function
#
# Now, let's define a function. The wrapped function can be synchronous or
# asynchronous, but calling it in either context will still work.
# Let's stick to a normal synchronous function


@stub.function
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


async def run_async():
    async with stub.run():
        # Call the function directly.
        print(await f(1000))

        # Parallel map.
        total = 0
        async for ret in f.map(range(20)):
            total += ret

        print(total)


# ## Entrypoint
#
# Trigger the async function. This is some boiler plate required by
# asyncio to get an event loop running and run the main function.

if __name__ == "__main__":
    asyncio.run(run_async())
