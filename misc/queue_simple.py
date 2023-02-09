# ---
# cmd: ["python", "-m", "misc.queue_simple"]
# ---
#
# # Using a queue to send/receive data
#
# This is an example of how to use queues to send/receive data.
# We don't do it here, but you could imagine doing this _between_ two functions.


import asyncio

import modal.aio

aio_stub = modal.aio.AioStub("example-queue-simple", q=modal.aio.AioQueue())


async def run_async():
    async with aio_stub.run() as app:
        q = app.q
        await q.put(42)
        r = await q.get()
        assert r == 42
        await q.put_many([42, 43, 44, 45, 46])
        await q.put_many([47, 48, 49, 50, 51])
        r = await q.get_many(3)
        assert r == [42, 43, 44]
        r = await q.get_many(99)
        assert r == [45, 46, 47, 48, 49, 50, 51]


async def many_consumers():
    async with aio_stub.run() as app:
        q = app.q
        print("Creating getters")
        tasks = [asyncio.create_task(q.get()) for i in range(20)]
        print("Putting values")
        await q.put_many(list(range(10)))
        await asyncio.sleep(1)
        # About 10 tasks should now be done
        n_done_tasks = sum(1 for t in tasks if t.done())
        assert n_done_tasks == 10
        # Finish remaining ones
        await q.put_many(list(range(10)))
        await asyncio.sleep(1)
        assert all(t.done() for t in tasks)


if __name__ == "__main__":
    asyncio.run(run_async())
    asyncio.run(many_consumers())
