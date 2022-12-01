import asyncio

import modal.aio

stub = modal.aio.AioStub("example-generators-async")


@stub.function
def f(i):
    for j in range(i):
        yield j


async def run_async():
    async with stub.run():
        async for r in f(10):
            print(r)

        async for r in f.map(range(5)):
            print(r)


if __name__ == "__main__":
    coro = run_async()
    asyncio.run(coro)
