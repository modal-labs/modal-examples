"""Load the small task subset used in the Qwen3.5-9B validation run."""

import json
from pathlib import Path
from urllib.request import urlopen

DATA_URL = (
    "https://raw.githubusercontent.com/NVIDIA-NeMo/ProRL-Agent-Server/"
    "6a1ead6bfac054fce6c1e62d1a77b330d96c58db/"
    "examples/swegym_slime_grpo/swegym_train_293.jsonl"
)


def load_tasks():
    with urlopen(DATA_URL, timeout=60) as response:
        rows = [json.loads(line) for line in response if line.strip()]
    instances = {
        row["metadata"]["instance_id"]: row["metadata"]["instance"] for row in rows
    }
    ids = json.loads(Path(__file__).with_name("task_ids.json").read_text())
    return [instances[instance_id] for instance_id in ids]
