# TODO: clean this up
import os
import subprocess
import sys
import time

from . import utils

MINUTES = 60
TIMEOUT = 20 * MINUTES


def run_script(example):
    t0 = time.time()

    try:
        print(f"cli args: {example.cli_args}")
        process = subprocess.run(
            example.cli_args,
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


def run_single_example(stem):
    examples = utils.get_examples()
    for example in examples:
        if stem == example.stem:
            run_script(example)
            break
    else:
        print(f"Could not find example name {stem}")
        exit(1)


if __name__ == "__main__":
    SystemExit(run_single_example(sys.argv[1]))
