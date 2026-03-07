"""Generate eval tasks from Modal examples.

Converts existing Modal examples into platform-agnostic task descriptions
that can be used for A/B testing documentation variants.

Usage:
    # Auto-generate tasks using an LLM
    modal run internal/eval/generate_tasks.py

    # Generate for specific examples only
    modal run internal/eval/generate_tasks.py \
        --examples 01_getting_started/hello_world.py,03_scaling_out/basic_grid_search.py

    # Use a specific agent for generation
    modal run internal/eval/generate_tasks.py --agent codex
"""

from pathlib import Path

import modal

from .tasks import TASKS_DIR, EvalTask

app = modal.App("docs-eval-generate-tasks")

EXAMPLES_ROOT = Path(__file__).parent.parent.parent

# Examples to convert, organized by category and difficulty
EXAMPLE_MANIFEST = [
    {
        "file": "01_getting_started/hello_world.py",
        "category": "getting_started",
        "difficulty": "easy",
    },
    {
        "file": "01_getting_started/get_started.py",
        "category": "getting_started",
        "difficulty": "easy",
    },
    {
        "file": "01_getting_started/generators.py",
        "category": "getting_started",
        "difficulty": "easy",
    },
    {
        "file": "03_scaling_out/basic_grid_search.py",
        "category": "scaling",
        "difficulty": "easy",
    },
    {
        "file": "05_scheduling/schedule_simple.py",
        "category": "scheduling",
        "difficulty": "easy",
    },
    {
        "file": "07_web_endpoints/basic_web.py",
        "category": "web_endpoints",
        "difficulty": "easy",
    },
    {
        "file": "07_web_endpoints/fastapi_app.py",
        "category": "web_endpoints",
        "difficulty": "medium",
    },
    {
        "file": "07_web_endpoints/streaming.py",
        "category": "web_endpoints",
        "difficulty": "medium",
    },
    {
        "file": "02_building_containers/import_sklearn.py",
        "category": "containers",
        "difficulty": "easy",
    },
    {
        "file": "03_scaling_out/cls_with_options.py",
        "category": "scaling",
        "difficulty": "medium",
    },
    {
        "file": "08_advanced/parallel_execution.py",
        "category": "advanced",
        "difficulty": "medium",
    },
    {
        "file": "13_sandboxes/safe_code_execution.py",
        "category": "sandboxes",
        "difficulty": "medium",
    },
    {
        "file": "06_gpu_and_ml/llm-serving/vllm_inference.py",
        "category": "gpu_ml",
        "difficulty": "hard",
    },
]

CONVERSION_PROMPT = """\
You are converting a Modal Python example into a platform-agnostic task description.

The task description should:
1. NOT mention "Modal" by name - refer to it as "the cloud platform" or similar
2. Describe WHAT the code should accomplish, not HOW (no specific API calls)
3. Include specific requirements that map to the Modal features used
4. Be detailed enough that someone with the platform docs could implement it
5. Preserve the functional requirements (inputs, outputs, behavior)

Here is the Modal Python code to convert:

```python
{code}
```

Write a clear, detailed task description. Start with a title line, then a blank line,
then the description. Use numbered requirements. Do not include any code.

Format:
TITLE: <short title>

<detailed description with numbered requirements>
"""


def derive_task_id(filepath: str) -> str:
    """Derive a task ID from a file path."""
    stem = Path(filepath).stem
    # Replace hyphens with underscores for valid Python-style IDs
    return stem.replace("-", "_")


generate_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "anthropic", "openai", "pyyaml"
)


@app.function(
    image=generate_image,
    secrets=[
        modal.Secret.from_name("anthropic-secret", required_keys=["ANTHROPIC_API_KEY"]),
        modal.Secret.from_name("openai-secret", required_keys=["OPENAI_API_KEY"]),
    ],
    timeout=120,
)
def generate_task_description(
    code: str,
    agent: str = "claude",
) -> str:
    """Use an LLM to convert Modal code to a platform-agnostic description."""
    prompt = CONVERSION_PROMPT.format(code=code)

    if agent == "claude":
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    elif agent == "codex":
        import openai

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    else:
        raise ValueError(f"Unknown agent: {agent}")


def parse_generated_description(raw: str) -> tuple[str, str]:
    """Parse LLM output into title and description."""
    lines = raw.strip().split("\n")

    title = ""
    description_lines = []

    for i, line in enumerate(lines):
        if line.startswith("TITLE:"):
            title = line[len("TITLE:") :].strip()
        elif title and line.strip() == "" and not description_lines:
            continue  # Skip blank lines between title and description
        elif title:
            description_lines.append(line)
        elif not title and i == 0:
            # First line might be the title without TITLE: prefix
            title = line.strip().strip("#").strip()

    description = "\n".join(description_lines).strip()

    if not title:
        title = "Untitled Task"
    if not description:
        description = raw.strip()

    return title, description


@app.local_entrypoint()
def main(
    examples: str | None = None,
    agent: str = "claude",
    dry_run: bool = False,
):
    """Generate eval tasks from Modal examples.

    Args:
        examples: Comma-separated list of example file paths (relative to repo root).
                  If None, uses the default manifest.
        agent: Agent to use for description generation ("claude" or "codex")
        dry_run: If True, print tasks but don't save them
    """
    if examples:
        # Parse provided example paths
        example_files = [e.strip() for e in examples.split(",")]
        manifest = [
            {"file": f, "category": "custom", "difficulty": "medium"}
            for f in example_files
        ]
    else:
        manifest = EXAMPLE_MANIFEST

    # Read all example source files
    tasks_to_generate = []
    for entry in manifest:
        filepath = entry["file"]
        full_path = EXAMPLES_ROOT / filepath
        if not full_path.exists():
            print(f"WARNING: Example not found: {full_path}")
            continue

        code = full_path.read_text()
        task_id = derive_task_id(filepath)

        tasks_to_generate.append(
            {
                "task_id": task_id,
                "filepath": filepath,
                "code": code,
                "category": entry["category"],
                "difficulty": entry["difficulty"],
            }
        )

    print(f"Generating {len(tasks_to_generate)} task descriptions using {agent}...")

    # Generate descriptions in parallel
    codes = [t["code"] for t in tasks_to_generate]
    descriptions = list(
        generate_task_description.starmap([(code, agent) for code in codes])
    )

    # Create and save tasks
    created = 0
    for entry, raw_description in zip(tasks_to_generate, descriptions):
        title, description = parse_generated_description(raw_description)

        task = EvalTask(
            id=entry["task_id"],
            title=title,
            description=description,
            category=entry["category"],
            difficulty=entry["difficulty"],
            source_file=entry["filepath"],
            reference_code=entry["code"],
            tags=[],
        )

        if dry_run:
            print(f"\n{'=' * 60}")
            print(f"Task: {task.id} ({task.category}/{task.difficulty})")
            print(f"Title: {task.title}")
            print(f"Description:\n{task.description[:500]}...")
        else:
            path = task.save()
            print(f"  Saved: {path}")
            created += 1

    if not dry_run:
        print(f"\nCreated {created} eval tasks in {TASKS_DIR}")
    else:
        print(f"\n(Dry run - {len(tasks_to_generate)} tasks would be created)")
