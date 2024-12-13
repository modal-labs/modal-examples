import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple, Optional

from utils import ExampleType, get_examples


class DeployError(NamedTuple):
    stdout: str
    stderr: str
    code: int


def deploy(
    deployable: bool,
    module_with_app: Path,
    dry_run: bool,
    filter_pttrn: Optional[str],
    env: Optional[dict[str, str]],
) -> Optional[DeployError]:
    if filter_pttrn and not re.match(filter_pttrn, module_with_app.name):
        return None

    if not deployable:
        print(f"⏩ skipping: '{module_with_app.name}' is not marked for deploy")
        return None

    deploy_command = f"modal deploy {module_with_app.name}"
    if dry_run:
        print(f"🌵  dry-run: '{module_with_app.name}' would have deployed")
    else:
        print(f"⛴ deploying: '{module_with_app.name}' ...")
        r = subprocess.run(
            shlex.split(deploy_command),
            cwd=module_with_app.parent,
            capture_output=True,
            env=os.environ | (env or {}),
        )
        if r.returncode != 0:
            print(
                f"⚠️ deployment failed: '{module_with_app.name}'",
                file=sys.stderr,
            )
            print(r.stderr)
            return DeployError(
                stdout=r.stdout, stderr=r.stderr, code=r.returncode
            )
        else:
            print(f"✔️ deployed '{module_with_app.name}")
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deploy Modal example programs to our Modal organization.",
        add_help=True,
    )
    parser.add_argument(
        "--dry-run",
        default=True,
        action="store_true",
        help="show what apps be deployed without deploying them.",
    )
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.add_argument(
        "--filter",
        default=None,
        help="Filter which apps are deployed with basic pattern matching. eg. 'cron' matches 'say_hello_cron.py'.",
    )
    arguments = parser.parse_args()

    if arguments.dry_run:
        print(
            "INFO: dry-run is active. Intended deployments will be displayed to console."
        )

    example_modules = (
        ex for ex in get_examples() if ex.type == ExampleType.MODULE
    )
    filter_pttrn = (
        (r".*" + arguments.filter + r".*") if arguments.filter else None
    )
    results = [
        deploy(
            deployable=bool(ex_mod.metadata.get("deploy")),
            module_with_app=Path(ex_mod.module),
            dry_run=arguments.dry_run,
            filter_pttrn=filter_pttrn,
            env=ex_mod.metadata.get("env"),
        )
        for ex_mod in example_modules
    ]

    failures = [r for r in results if r]
    if any(failures):
        print(f"ERROR: {len(failures)} deployment failures.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
