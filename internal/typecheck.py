"""
MyPy type-checking script.
Unvalidated, incorrect type-hints are worse than no type-hints!
"""

import pathlib
import subprocess
import sys

import mypy.api


def fetch_git_repo_root() -> pathlib.Path:
    return pathlib.Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("ascii")
        .strip()
    )


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
    repo_root = fetch_git_repo_root()
    config_file = repo_root / "pyproject.toml"
    errors = []

    # Type-check scripts:
    # (Only finds numbered folders up until '99_*')
    topic_dirs = sorted(
        [d for d in repo_root.iterdir() if d.name[:2].isdigit()]
    )
    for topic_dir in topic_dirs:
        # Most topic directories have only independent .py module files.
        # But in some places topic directories have subdirectory packages, which
        # are independent examples and should be type-checked independently.
        #
        # Ignore any non-Python files.
        #
        # TODO: parallelize type-checking across packages.
        for pth in topic_dir.iterdir():
            if (
                pth.is_file() and not pth.name.endswith(".py")
            ) or pth.name == "__pycache__":
                continue
            print(
                f"⌛️ running mypy on '{topic_dir.name}/{pth.name}'",
                file=sys.stderr,
            )
            topic_errors = extract_errors(
                run_mypy(
                    pkg=str(pth),
                    config_file=config_file,
                )
            )
            if topic_errors:
                print("\n".join(topic_errors))
                errors.extend(topic_errors)

    # Type-check packages:
    # Getting mypy running successfully with a monorepo of heterogenous packaging structures
    # is a bit fiddly, so we expect top-level packages to opt-in to type-checking by placing a
    # `py.typed` file inside themselves. https://peps.python.org/pep-0561/
    for py_typed in repo_root.glob("**/py.typed"):
        toplevel_pkg = py_typed.parent
        print(f"⌛️ running mypy on '{toplevel_pkg}'", file=sys.stderr)
        package_errors = extract_errors(
            run_mypy(
                toplevel_pkg=str(toplevel_pkg),
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
