# ---
# cmd: ["python", "-m", "05_scheduling.schedule_simple"]
# ---

# # Scheduling functions
# This example shows how you can schedule functions to run at specific times.
# There are two different ways of doing this with Modal.
# We
# define two functions that print the current time. The first function runs every
# 5 seconds, and the second function runs every minute. We use the `schedule`
# argument to specify the schedule for each function. The `schedule` argument can
# take a `modal.Period` object to specify a time interval or a `modal.Cron` object
# to specify a cron expression. We run the app for 60 seconds to see the functions
# in action.

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
