"""Eval task data model and loading utilities."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

TASKS_DIR = Path(__file__).parent / "tasks"


@dataclass
class EvalTask:
    """A single evaluation task derived from a Modal example."""

    id: str
    title: str
    description: str  # Platform-agnostic task description
    category: str
    difficulty: str  # easy, medium, hard
    source_file: str  # Path to original example (relative to repo root)
    reference_code: str  # Original Modal code for comparison
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalTask":
        return cls(**data)

    def save(self, path: Path | None = None) -> Path:
        """Save task to YAML file."""
        if path is None:
            path = TASKS_DIR / f"{self.id}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        return path

    @classmethod
    def load(cls, path: Path) -> "EvalTask":
        """Load task from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


def load_all_tasks(
    task_dir: Path | None = None,
    task_ids: list[str] | None = None,
    categories: list[str] | None = None,
) -> list[EvalTask]:
    """Load all tasks from the tasks directory, optionally filtering."""
    if task_dir is None:
        task_dir = TASKS_DIR

    tasks = []
    for yaml_file in sorted(task_dir.glob("*.yaml")):
        task = EvalTask.load(yaml_file)

        if task_ids and task.id not in task_ids:
            continue
        if categories and task.category not in categories:
            continue

        tasks.append(task)

    return tasks


@dataclass
class EvalResult:
    """Result of evaluating a single (task, docs_variant, agent) combination."""

    task_id: str
    docs_variant: str
    agent: str
    model: str
    generated_code: str
    scores: dict  # Individual scores
    overall_score: float  # Aggregate score (0-1)
    error: str | None = None  # Error message if generation failed

    def to_dict(self) -> dict:
        return asdict(self)


def load_docs_variant(variant: str, docs_dir: Path | None = None) -> str:
    """Load documentation content for a given variant.

    A variant can be:
    - A name matching a directory under docs/ (e.g., "llms_txt")
    - A URL to fetch (e.g., "https://modal.com/llms.txt")
    - A file path
    """
    if docs_dir is None:
        docs_dir = Path(__file__).parent / "docs"

    variant_dir = docs_dir / variant
    if variant_dir.is_dir():
        # Concatenate all files in the variant directory
        parts = []
        for f in sorted(variant_dir.iterdir()):
            if f.is_file() and f.suffix in (".txt", ".md"):
                parts.append(f.read_text())
        return "\n\n---\n\n".join(parts)

    variant_file = docs_dir / variant
    if variant_file.is_file():
        return variant_file.read_text()

    # Try as absolute path
    variant_path = Path(variant)
    if variant_path.is_file():
        return variant_path.read_text()

    raise FileNotFoundError(
        f"Docs variant '{variant}' not found. "
        f"Looked in: {variant_dir}, {variant_file}, {variant_path}"
    )


def fetch_docs_from_url(url: str) -> str:
    """Fetch documentation content from a URL."""
    import urllib.request

    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")
