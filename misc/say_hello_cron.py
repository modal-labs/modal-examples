# # Deploy a cron job with Modal

# This example shows how you can deploy a cron job with Modal.

import time
from datetime import datetime, timezone

import modal

app = modal.App("example-say-hello-cron")


@app.function(schedule=modal.Period(seconds=10))
def say_hello():
    start_time = datetime.now(timezone.utc)
    for i in range(10):
        print(f"Message #{i} from invocation at {start_time}")
        time.sleep(1.5)
