# ---
# cmd: ["python", "-m", "05_scheduling.schedule_simple"]
# ---

# # Scheduling remote jobs

# This example shows how you can schedule remote jobs on Modal.
# You can do this either with:
#
# - [`modal.Period`](https://modal.com/docs/reference/modal.Period) - a time interval between function calls.
# - [`modal.Cron`](https://modal.com/docs/reference/modal.Cron) - a cron expression to specify the schedule.

# In the code below, the first function runs every
# 5 seconds, and the second function runs every minute. We use the `schedule`
# argument to specify the schedule for each function. The `schedule` argument can
# take a `modal.Period` object to specify a time interval or a `modal.Cron` object
# to specify a cron expression.

import time
from datetime import datetime

import modal

app = modal.App("example-schedule-simple")


@app.function(schedule=modal.Period(seconds=5))
def print_time_1():
    print(
        f'Printing with period 5 seconds: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}'
    )


@app.function(schedule=modal.Cron("* * * * *"))
def print_time_2():
    print(
        f'Printing with cron every minute: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}'
    )


if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            time.sleep(60)
