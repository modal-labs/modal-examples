"""
MyPy type-checking script.
Unvalidated, incorrect type-hints are worse than no type-hints!
"""

import concurrent
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import mypy.api


def fetch_examples_root() -> pathlib.Path:
    # Anchor on this file rather than the git repository root, so the script
    # also works when the examples live in a subdirectory of a larger
    # repository.
    return pathlib.Path(__file__).resolve().parent.parent


def run_mypy(pkg: str, config_file: pathlib.Path) -> list[str]:
    args = [
        pkg,
        "--no-incremental",
        "--namespace-packages",
        "--config-file",
        str(config_file),
    ]
    result = mypy.api.run(args)
    return result[0].splitlines()


def extract_errors(output: list[str]) -> list[str]:
    if len(output) > 0 and "success" in output[0].lower():
        print(output[0], file=sys.stderr)
        return []
    return [l for l in output if "error" in l]


def main() -> int:
    examples_root = fetch_examples_root()
    config_file = examples_root / "pyproject.toml"
    errors = []

    # Type-check scripts
    topic_dirs = sorted([d for d in examples_root.iterdir() if d.name[:2].isdigit()])

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_path = {}
        for topic_dir in topic_dirs:
            for pth in topic_dir.iterdir():
                if not (pth.is_file() and pth.name.endswith(".py")):
                    continue
                elif "__pycache__" in pth.parts:
                    continue
                else:
                    print(f"⌛️ spawning mypy on '{pth}'", file=sys.stderr)
                    future = executor.submit(
                        run_mypy, pkg=str(pth), config_file=config_file
                    )
                    future_to_path[future] = pth

        # The timeout is a total budget for all files; it needs headroom for
        # low-core-count CI runners, where few mypy processes run in parallel.
        for future in concurrent.futures.as_completed(future_to_path, timeout=600):
            pth = future_to_path[future]
            try:
                output = future.result()
                topic_errors = extract_errors(output)
                if topic_errors:
                    print(f"\nfound {len(topic_errors)} errors in '{pth}'")
                    print("\n".join(topic_errors))
                    errors.extend(topic_errors)
            except Exception as exc:
                print(f"Error on file {pth}: {exc}")
                errors.append(exc)

    # Type-check packages
    # Getting mypy running successfully with a monorepo of heterogenous packaging structures
    # is a bit fiddly, so we expect top-level packages to opt-in to type-checking by placing a
    # `py.typed` file inside themselves. https://peps.python.org/pep-0561/
    for py_typed in examples_root.glob("**/py.typed"):
        if "site-packages" in py_typed.parts:
            continue
        toplevel_pkg = py_typed.parent
        print(f"⌛️ running mypy on '{toplevel_pkg}'", file=sys.stderr)
        package_errors = extract_errors(
            run_mypy(
                pkg=str(toplevel_pkg),
                config_file=config_file,
            )
        )
        if package_errors:
            print(
                f"found {len(package_errors)} errors in '{toplevel_pkg}'",
                file=sys.stderr,
            )
            print("\n".join(package_errors))
            errors.extend(package_errors)

    if errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
