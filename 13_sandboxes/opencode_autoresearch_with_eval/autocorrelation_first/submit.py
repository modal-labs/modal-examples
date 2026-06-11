import argparse
import importlib.util
import json
import numbers
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any


def load_solution(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("candidate_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return float(value)
    if value is None or isinstance(value, str | bool):
        return value
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def run_candidate(solution_path: Path) -> dict[str, Any]:
    module = load_solution(solution_path)
    if not hasattr(module, "run_code"):
        raise RuntimeError("solution.py must define run_code()")

    output = module.run_code()
    if isinstance(output, tuple | list):
        solution = output[0]
        reported = output[1] if len(output) > 1 else None
    else:
        solution = output
        reported = None

    return {
        "solution": to_jsonable(solution),
        "reported": to_jsonable(reported),
    }


def post_submission(grader_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{grader_url.rstrip('/')}/submit",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode())


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit solution.py to the verifier")
    parser.add_argument("--solution", default="solution.py", help="Candidate file")
    parser.add_argument(
        "--grader-url",
        default=os.environ.get("GRADER_URL"),
        help="Verifier URL. Defaults to GRADER_URL.",
    )
    parser.add_argument("--message", default=None, help="Optional submission note")
    args = parser.parse_args()

    if not args.grader_url:
        raise SystemExit("Set GRADER_URL or pass --grader-url.")

    payload = run_candidate(Path(args.solution))
    if args.message:
        payload["message"] = args.message

    result = post_submission(args.grader_url, payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"submit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
