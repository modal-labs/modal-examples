# ---
# cmd: ["python", "13_sandboxes/sandbox_pool.py"]
# pytest: false
# ---

# # Build a pool of warm sandboxes that are healthy and ready to serve requests


import time

import modal

app = modal.App("sandbox-pool")

pool_queue = modal.Queue.from_name("sandbox-pool-buffer", create_if_missing=True)

server_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]",
    "requests",
)
sandbox_image = modal.Image.debian_slim()


SERVER_PORT = 8080
HEALTH_CHECK_TIMEOUT_SECONDS = 10
SANDBOX_TIMEOUT_SECONDS = 60 * 60
SANDBOX_USE_DURATION_SECONDS = 5 * 60
DEFAULT_POOL_SIZE = 2


def health_check_sandbox(daemon_url):
    import requests

    start_time = time.time()
    while time.time() - start_time < HEALTH_CHECK_TIMEOUT_SECONDS:
        try:
            response = requests.get(daemon_url, timeout=HEALTH_CHECK_TIMEOUT_SECONDS)
            response.raise_for_status()
            return
        except requests.RequestException:
            if time.time() - start_time >= HEALTH_CHECK_TIMEOUT_SECONDS:
                raise
            time.sleep(0.1)

    raise requests.RequestException("Health check failed")


@app.function(image=server_image)
@modal.concurrent(max_inputs=100)
def add_sandbox_to_queue():
    # This is done so we don't create sandboxes in the ephemeral app
    deployed_app = modal.App.lookup("sandbox-pool", create_if_missing=True)

    server_command = ["python", "-m", "http.server", f"{SERVER_PORT}"]
    sb = modal.Sandbox.create(
        *server_command,
        app=deployed_app,
        image=sandbox_image,
        encrypted_ports=[SERVER_PORT],
        timeout=SANDBOX_TIMEOUT_SECONDS,
    )
    expiration_time = int(time.time()) + SANDBOX_TIMEOUT_SECONDS
    url = sb.tunnels()[SERVER_PORT].url

    health_check_sandbox(url)

    pool_queue.put((sb.object_id, url, expiration_time))


@app.function(image=server_image)
@modal.fastapi_endpoint()
@modal.concurrent(max_inputs=100)
def get_sandbox() -> str:
    while res := pool_queue.get(block=True):
        sb_id, url, expiration_time = res

        # backfill + ensures the queue will have at least one sandbox
        add_sandbox_to_queue.spawn()

        if expiration_time < time.time() + SANDBOX_USE_DURATION_SECONDS:
            print(f"Sandbox {sb_id} does not have enough time left - removing it")
            sb = modal.Sandbox.from_id(sb_id)
            sb.terminate()
            continue

        return url

    raise RuntimeError("No sandbox with enough time left")


@app.local_entrypoint()
def resize_pool(target: int = DEFAULT_POOL_SIZE):
    if target < 0:
        raise ValueError("Target pool size must be non-negative")

    current_size = pool_queue.len()
    diff = target - current_size

    if diff > 0:
        for _ in add_sandbox_to_queue.starmap(() for _ in range(diff)):
            pass
    elif diff < 0:
        for _ in range(-diff):
            if res := pool_queue.get(block=False):
                sb_id, _, _ = res
                sb = modal.Sandbox.from_id(sb_id)
                sb.terminate()
            else:
                break

    print(f"Changed pool size by {diff:+d}, now at {pool_queue.len()} sandboxes.")


@app.local_entrypoint()
def check_pool(verbose: bool = False):
    print(f"Number of sandboxes in the pool: {pool_queue.len()}")
    if verbose:
        for sb_id, url, expiration_time in pool_queue.iterate():
            seconds_left = expiration_time - time.time()
            print(
                f"Sandbox '{sb_id}' is at {url} and expires at {expiration_time} "
                f"({seconds_left} seconds left)"
            )
