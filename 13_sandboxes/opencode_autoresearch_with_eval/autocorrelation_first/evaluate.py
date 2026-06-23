import math
import time
from pathlib import Path
from typing import Any

import fastapi
from pydantic import BaseModel

app = fastapi.FastAPI()

TASK_TEXT = Path("autocorrelation_first.txt").read_text()
MAX_SEQUENCE_LENGTH = 4096
MAX_STORED_SUBMISSIONS = 20
SUBMISSIONS: list[dict[str, Any]] = []


class Submission(BaseModel):
    solution: list[Any]
    reported: float | None = None
    message: str | None = None


def normalize_sequence(sequence: list[Any]) -> list[float]:
    if not sequence:
        raise ValueError("solution must not be empty")
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        raise ValueError(f"solution is too long; max length is {MAX_SEQUENCE_LENGTH}")

    normalized = []
    for value in sequence:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("solution values must be numbers")
        if not math.isfinite(value):
            raise ValueError("solution values must be finite")
        normalized.append(min(1000.0, max(0.0, float(value))))

    if sum(normalized) < 0.01:
        raise ValueError("solution mass is too small")

    return normalized


def c1_score(sequence: list[float]) -> float:
    convolution = [0.0] * (2 * len(sequence) - 1)
    for i, left in enumerate(sequence):
        for j, right in enumerate(sequence):
            convolution[i + j] += left * right

    return 2 * len(sequence) * max(convolution) / (sum(sequence) ** 2)


def build_record(payload: Submission) -> dict[str, Any]:
    submitted_at = time.time()
    try:
        solution = normalize_sequence(payload.solution)
        c1 = c1_score(solution)
        return {
            "valid": True,
            "c1": c1,
            "score": 1.0 / (1e-8 + c1),
            "length": len(solution),
            "reported": payload.reported,
            "message": payload.message,
            "solution": solution,
            "submitted_at": submitted_at,
        }
    except ValueError as exc:
        return {
            "valid": False,
            "c1": float("inf"),
            "score": 0.0,
            "length": len(payload.solution),
            "reported": payload.reported,
            "message": payload.message,
            "error": str(exc),
            "solution": payload.solution,
            "submitted_at": submitted_at,
        }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/task")
def task() -> dict[str, str]:
    return {"task": TASK_TEXT}


@app.post("/submit")
def submit(payload: Submission) -> dict[str, Any]:
    record = build_record(payload)
    SUBMISSIONS.append(record)
    SUBMISSIONS.sort(key=lambda item: item["c1"])
    del SUBMISSIONS[MAX_STORED_SUBMISSIONS:]
    return record


@app.get("/submissions")
def submissions() -> dict[str, Any]:
    valid = [record for record in SUBMISSIONS if record["valid"]]
    return {
        "task": TASK_TEXT,
        "best": valid[0] if valid else None,
        "submissions": SUBMISSIONS,
    }
