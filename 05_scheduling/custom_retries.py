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

# ## Use a Dict to track call count across retries
#
# Each retry runs in a new container invocation, so we use a
# [`modal.Dict`](https://modal.com/docs/reference/modal.Dict) to share
# state and make the demo deterministic.

call_counter = modal.Dict.from_name(
    "custom-retries-demo-counter", create_if_missing=True
)

# ## Demo App
#
# This function follows a scripted sequence to demonstrate the behavior:
#
# 1. **Call 1** — raises `TimeoutError` (retryable → Modal retries)
# 2. **Call 2** — raises `ConnectionError` (retryable → Modal retries)
# 3. **Call 3** — raises `MemoryError` (crashable → returned, no more retries)
#
# So you'll see two retries, then a clean stop on the third attempt.


@app.function(retries=modal.Retries(max_retries=5, initial_delay=1.0))
def flaky_task():
    call_count = call_counter.get("calls", 0) + 1
    call_counter["calls"] = call_count
    print(f"Attempt {call_count}")

    # Scripted error sequence
    errors = [
        TimeoutError("GPU timed out"),  # attempt 1: retryable
        ConnectionError("lost connection to data server"),  # attempt 2: retryable
        MemoryError("CUDA out of memory"),  # attempt 3: crashable
    ]
    error = errors[min(call_count, len(errors)) - 1]

    print(f"  Hit: {error!r}")

    if isinstance(error, retry_exceptions):
        print("  -> retryable, re-raising so Modal retries")
        raise error

    # Return instead of raise — Modal sees success, stops retrying
    print("  -> non-retryable, returning error to stop retries")
    return error


# ## Entrypoint
#
# The caller checks whether the return value is an exception.


@app.local_entrypoint()
def main():
    call_counter["calls"] = 0  # reset counter
    result = flaky_task.remote()
    if isinstance(result, Exception):
        print(f"Stopped with non-retryable error: {result!r}")
    else:
        print(f"Result: {result}")
