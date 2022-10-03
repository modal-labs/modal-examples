import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import jupytext
import jupytext.config

DEFAULT_DIRECTORY = Path(__file__).parent


class ExampleType(Enum):
    MODULE = 1
    ASSET = 2


@dataclass
class Example:
    type: ExampleType
    filename: str
    module: Optional[str]
    metadata: Optional[dict]


_RE_NEWLINE = re.compile(r"\r?\n")
_RE_FRONTMATTER = re.compile(r"^---$", re.MULTILINE)


def render_example_md(content: str, filename: str) -> str:
    """Render a Python code example to Markdown documentation format."""

    lines = _RE_NEWLINE.split(content)
    markdown: list[str] = []
    code: list[str] = []
    for line in lines:
        if line == "#" or line.startswith("# "):
            if code:
                markdown.extend(["```python", *code, "```", ""])
                code = []
            markdown.append(line[2:])
        else:
            markdown.append("")
            if code or line:
                code.append(line)

    if code:
        markdown.extend(["```python", *code, "```", ""])

    # hacky accounting for variable filename depth
    relative_path = filename[
        filename.index("modal/examples/") + len("modal/examples/") :
    ]
    github_url = (
        f"https://github.com/modal-labs/modal-examples/blob/main/{relative_path}"
    )
    markdown.append(
        f"\n_The raw source code for this example can be found [on GitHub]({github_url})._\n",
    )

    text = "\n".join(markdown)
    if _RE_FRONTMATTER.match(text):
        # Strip out frontmatter from text.
        if match := _RE_FRONTMATTER.search(text, 4):
            text = text[match.end() :]
    return text


def get_examples(directory: Path = DEFAULT_DIRECTORY):
    if not directory.exists():
        raise Exception(
            f"Can't find directory {directory}. You might need to clone the modal-examples repo there"
        )

    print(directory)
    config = jupytext.config.JupytextConfiguration(
        root_level_metadata_as_raw_cell=False
    )
    ignored = []
    for subdir in sorted(list(directory.iterdir())):
        if not subdir.is_dir():
            continue
        for filename in sorted(list(subdir.iterdir())):
            filename_abs: str = str(filename.resolve())
            ext: str = filename.suffix
            if ext == ".py":
                module = f"{subdir.stem}.{filename.stem}"
                data = jupytext.read(open(filename_abs), config=config)
                metadata = data["metadata"]["jupytext"].get("root_level_metadata", {})
                yield Example(ExampleType.MODULE, filename_abs, module, metadata)
            elif ext in [".png", ".jpeg", ".jpg", ".gif", ".mp4"]:
                yield Example(ExampleType.ASSET, filename_abs, None, None)
            else:
                ignored.append(str(filename))
    print(f"Ignoring examples files: {ignored}")


if __name__ == "__main__":
    for example in get_examples():
        print(example)
