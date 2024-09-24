# # Run async generator function on Modal

# This example shows how you can run an async generator function on Modal.
# Modal natively supports async/await syntax using asyncio.

import modal

app = modal.App("example-generators-async")


@app.function()
def f(i):
    for j in range(i):
        yield j


@app.local_entrypoint()
async def run_async():
    async for r in f.remote_gen.aio(10):
        print(r)
