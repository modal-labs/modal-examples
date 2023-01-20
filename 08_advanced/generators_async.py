import modal.aio

stub = modal.aio.AioStub("example-generators-async")


@stub.function
def f(i):
    for j in range(i):
        yield j


@stub.local_entrypoint
async def run_async():
    async for r in f.call(10):
        print(r)

    async for r in f.map(range(5)):
        print(r)
