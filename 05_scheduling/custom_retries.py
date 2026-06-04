# ---
# cmd: ["modal", "run", "05_scheduling.custom_retries"]
# ---

# # Custom retries by exception type
# There are two types of retries in Modal:
# 1. When a function execution is interrupted by [preemption](https://modal.com/docs/guide/preemption#preemption), the input will be retried. This behavior is not configurable at this time.
# 2. When a function execution fails, ie by a raised Exception, Modal will retry the function call if you have [`modal.Retries`](https://modal.com/docs/reference/modal.Retries) configured.
# This example is about customizing the latter to only retry on certain exception types.
# For example, you may only want to retry on certain expected errors (e.g. timeouts, or
# transient network errors) and crash immediately on others (e.g. OOM, bad input).

# The trick is to:
# 1. Raise retryable errors in the usual way to trigger `modal.Retries`
# 2. Catch and `return` non-retryable errors.
# For #2, Modal will see a successful function call execution and return the exception
# to your client/server to handle as desired.

import modal

app = modal.App("example-custom-retries")

# ## Define retryable vs. crashable exceptions

retry_exceptions = (
    TimeoutError,
    ConnectionError,
    # transient CUDA errors, network blips, etc.
)

crashable_exceptions = (
    MemoryError,
    ValueError,
    # OOM, bad input — retrying won't help
)

# ## Demo App
#
# Each retry runs in a fresh container, so we drive the demo with a
# [`modal.Dict`](https://modal.com/docs/reference/modal.Dict) keyed by attempt
# index. Each invocation pops the next scripted error and either re-raises it
# (retryable → Modal retries) or returns it (crashable → Modal stops).


@app.function(retries=modal.Retries(max_retries=5, initial_delay=1.0))
def flaky_task(errors):
    attempt = min(errors.keys())  # lowest index not yet handled
    error = errors.pop(attempt)
    print(f"Attempt {attempt}: hit {error!r}")

    if isinstance(error, retry_exceptions):
        raise error  # re-raise so Modal retries with the same Dict
    return error  # return so Modal sees success and stops retrying


# ## Entrypoint
#
# We stage the scripted errors in an [ephemeral `Dict`](https://modal.com/docs/reference/modal.Dict)
# that lives only for the run, pass it in, then check whether the result is an
# exception:
#
# 1. **Attempt 0** — `TimeoutError` (retryable → Modal retries)
# 2. **Attempt 1** — `ConnectionError` (retryable → Modal retries)
# 3. **Attempt 2** — `MemoryError` (crashable → returned, no more retries)


@app.local_entrypoint()
def main():
    errors = {
        0: TimeoutError("GPU timed out"),
        1: ConnectionError("lost connection to data server"),
        2: MemoryError("CUDA out of memory"),
    }
    with modal.Dict.ephemeral() as state:
        state.update(errors)
        result = flaky_task.remote(state)

    if isinstance(result, Exception):
        print(f"Stopped with non-retryable error: {result!r}")
    else:
        print(f"Result: {result}")
