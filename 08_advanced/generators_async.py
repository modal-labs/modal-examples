import modal

stub = modal.Stub("example-generators-async")


@stub.function()
def f(i):
    for j in range(i):
        yield j


@stub.local_entrypoint()
async def run_async():
    async for r in f.call.aio(10):
        print(r)

    async for r in f.map.aio(range(5)):
        print(r)
