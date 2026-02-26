# # Sticky routing for Modal HTTP Servers

# This example demonstrates the usage and behavior of
# the optional "sticky" routing behavior of
# Modal HTTP Servers with a basic routing test.

# For a gentler introduction to Modal HTTP Servers,
# see [this example](https://modal.com/docs/examples/http_server).

# In sticky routing, sequential requests from the same client
# are sent to the same server replica.
# Modal HTTP Servers offer sticky routing for fixed replica sets
# using [rendezvous hashing](https://randorithms.com/2020/12/26/rendezvous-hashing.html),
# ensuring that as your servers scale up and down, load stays balanced across replicas
# and clients are typically routed to the same replica for repeated requests.

# Note that requests are not _guaranteed_ to be routed to the same replica,
# and so this form of sticky routing should not be relied on for logical correctness.
# Instead, this sticky routing is intended to be used as a performance optimization,
# as in KV cacheing for [Transformer LLM inference](https://modal.com/docs/examples/sglang_low_latency).

# ## Define the Modal HTTP Server

# First, we import the libraries we'll use both locally, to run a routing test,
# and remotely, to run our server.

# We also define our Modal [App](https://modal.com/docs/guide/apps)
# and the Modal [Image](https://modal.com/docs/guide/images)
# that provides the dependencies of our server code.

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
import modal
import modal.experimental
from rich.console import Console


app = modal.App("example-http-server-sticky")

image = modal.Image.debian_slim().uv_pip_install("fastapi[standard]==0.115.4")

# Now we can define our HTTP Server.
# We set the minimum number of containers (replicas)
# to be greater than one so that there are multiple
# replicas available for routing during our test.

# Additionally, we set the regions into which we
# want to deploy the proxies that communicate between
# our clients and the server.

# We also use the [`modal.concurrent` decorator](https://modal.com/docs/guide/concurrent-inputs)
# to allow each HTTP Server replica to handle more than one input.

# Modal HTTP Servers are structured as Modal [Clses](https://modal.com/docs/guide/lifecycle-functions)
# that start a process or thread that listens on the provided `port` in a `modal.enter`-decorated method.
# Here, we spin up a simple FastAPI server that returns the
# [identity of the replica within Modal](https://modal.com/docs/guide/environment_variables)
# and run it with `uvicorn`.

PORT = 8000
CONTAINERS = 2
PROXY_REGIONS = ["us-west"]


@app.cls(image=image, min_containers=CONTAINERS)
@modal.experimental.http_server(port=PORT, proxy_regions=PROXY_REGIONS)
@modal.concurrent(target_inputs=100)
class Server:
    @modal.enter()
    def start(self):
        import os
        import threading

        import uvicorn
        from fastapi import FastAPI

        container_id = os.environ["MODAL_TASK_ID"]
        fastapi_app = FastAPI(title=container_id)

        @fastapi_app.post("/")
        async def whoami():
            return {"CONTAINER_ID": container_id}

        self.thread = threading.Thread(
            target=uvicorn.run,
            kwargs={"app": fastapi_app, "host": "0.0.0.0", "port": PORT},
            daemon=True,
        )
        self.thread.start()


# ## Test the routing behavior of the Modal HTTP Server


@app.local_entrypoint()
async def test(n_clients: int = 10, sticky: bool = True, seconds: float = 5.0):

    url = (await Server._experimental_get_flash_urls.aio())[0]
    async with aiohttp.ClientSession() as sess:
        await wait_available(sess, url)

    # allow generous time for all replicas to spin up;
    # remove this sleep and increase CONTAINERS
    # to observe session routing changes during autoscaling
    await asyncio.sleep(5 + ((CONTAINERS - 10) // 2))

    # run the test
    results = await run_clients(url, n_clients, seconds, sticky)
    stats = aggregate_results(results)

    # give time for server logs to flush,
    await asyncio.sleep(1)
    # then display results
    print_summary(url, sticky, n_clients, seconds, stats)

    if sticky and stats["multi"]:
        raise AssertionError("Sticky routing violated for some clients")


# ```bash
# modal run http_server_sticky.py
# ```

# ## Write the client for the Modal HTTP Server


async def wait_available(sess: aiohttp.ClientSession, url: str) -> None:
    while True:
        async with sess.post(url, json={}) as resp:
            if resp.status != 503:
                return


@dataclass
class ClientResult:
    client_id: int
    containers_seen: set[str]
    requests_ok: int
    requests_err: int


async def client(
    url: str, client_id: int, seconds: float, sticky: bool
) -> ClientResult:
    headers = {"Modal-Session-Id": str(client_id)} if sticky else {}
    end = time.monotonic() + seconds

    seen: set[str] = set()
    n_ok: int = 0
    n_err: int = 0

    async with aiohttp.ClientSession(headers=headers) as sess:
        while time.monotonic() < end:
            async with sess.post(
                url, json={}, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    seen.add(data["CONTAINER_ID"])
                    n_ok += 1
                else:
                    n_err += 1

    return ClientResult(client_id, seen, n_ok, n_err)


# ## Addenda

# The remainder of this code is required for this example to run
# but is not necessary for Modal HTTP Servers or their clients in general.
# For instance, it defines the logic for concurrency and result aggregation/display
# for this particular routing test.


async def run_clients(
    url: str, n_clients: int, seconds: float, sticky: bool
) -> list[ClientResult]:
    tasks = [client(url, c, seconds, sticky) for c in range(n_clients)]
    return list(await asyncio.gather(*tasks))


def aggregate_results(results: list[ClientResult]) -> dict[str, Any]:
    total_ok = sum(r.requests_ok for r in results)
    total_err = sum(r.requests_err for r in results)
    multi = {
        r.client_id: r.containers_seen for r in results if len(r.containers_seen) > 1
    }

    per_client = [(r.client_id, r.containers_seen) for r in results]

    return {
        "total_ok": total_ok,
        "total_err": total_err,
        "multi": multi,
        "per_client": per_client,
    }


def print_summary(
    url: str,
    sticky: bool,
    n_clients: int,
    seconds: float,
    stats: dict[str, Any],
    console: Console | None = None,
) -> None:
    if not console:
        console = Console()
    console.print()
    console.print(
        f"[bold]url=[/]{url} [bold]sticky=[/]{sticky} [bold]clients=[/]{n_clients} [bold]duration_s=[/]{seconds}"
    )
    console.print(
        f"[green]total_ok={stats['total_ok']}[/] [red]total_err={stats['total_err']}[/]"
    )

    for c, seen in stats["per_client"]:
        console.print(f"  client={c} containers={list(seen)}")
    console.print(
        f"Clients with multiple containers: [yellow]{len(stats['multi'])}/{n_clients}[/]"
    )
