import os
import subprocess
import sys
import time

MINUTES = 60
TIMEOUT = 20 * MINUTES


def run_script(cli_args):
    t0 = time.time()

    try:
        print(f"cli args: {cli_args}")
        process = subprocess.run(
            cli_args,
            env=os.environ,
            capture_output=False,
            timeout=TIMEOUT,
            stderr=sys.stderr,
            stdout=sys.stdout,
        )
        total_time = time.time() - t0
        if process.returncode == 0:
            print(f"Success after {total_time:.2f}s :)")
        else:
            print(
                f"Failed after {total_time:.2f}s with return code {process.returncode} :("
            )

        returncode = process.returncode

    except subprocess.TimeoutExpired:
        print(f"Past timeout of {TIMEOUT}s :(")
        returncode = 999

    return returncode


if __name__ == "__main__":
    SystemExit(run_script(sys.argv))
