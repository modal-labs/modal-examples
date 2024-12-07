import os
import subprocess
import sys
import time

from . import utils

MINUTES = 60
TIMEOUT = 12 * MINUTES


def run_script(example):
    t0 = time.time()

    try:
        print(f"cli args: {example.cli_args}")
        process = subprocess.run(
            map(str, example.cli_args),
            env=os.environ | example.env | {"MODAL_SERVE_TIMEOUT": "5.0"},
            timeout=TIMEOUT,
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
        if stem == example.stem and example.metadata.get("lambda-test", True):
            return run_script(example)
    else:
        print(f"Could not find example name {stem}")
        return 0


if __name__ == "__main__":
    sys.exit(run_single_example(sys.argv[1]))
