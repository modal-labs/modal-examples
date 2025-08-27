import argparse
import os
import random
import subprocess
import sys
import time

from . import utils

MINUTES = 60
DEFAULT_TIMEOUT = 12 * MINUTES


def run_script(example, timeout=DEFAULT_TIMEOUT):
    t0 = time.time()

    print(f"Running example {example.stem} with timeout {timeout}s")

    try:
        print(f"cli args: {example.cli_args}")
        if "runc" in example.runtimes:
            example.env |= {"MODAL_FUNCTION_RUNTIME": "runc"}
        process = subprocess.run(
            [str(x) for x in example.cli_args],
            env=os.environ | example.env | {"MODAL_SERVE_TIMEOUT": "5.0"},
            timeout=timeout,
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
        print(f"Past timeout of {timeout}s :(")
        returncode = 999

    return returncode


def run_single_example(stem, timeout=DEFAULT_TIMEOUT):
    examples = utils.get_examples()
    for example in examples:
        if stem == example.stem and example.metadata.get("lambda-test", True):
            return run_script(example, timeout=timeout)
    else:
        print(f"Could not find example name {stem}")
        return 0


def run_random_example(timeout=DEFAULT_TIMEOUT):
    examples = filter(
        lambda ex: ex.metadata and ex.metadata.get("lambda-test", True),
        utils.get_examples(),
    )
    return run_script(random.choice(list(examples)), timeout=timeout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("example", nargs="?", default=None)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()
    print(args)
    if args.example:
        sys.exit(run_single_example(args.example, timeout=args.timeout))
    else:
        sys.exit(run_random_example(timeout=args.timeout))
