import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import jupytext
import jupytext.config

DEFAULT_DIRECTORY = os.path.dirname(__file__)


class ExampleType(Enum):
    MODULE = 1
    ASSET = 2


@dataclass
class Example:
    type: ExampleType
    filename: str
    module: Optional[str]
    metadata: Optional[dict]


def get_examples(directory=DEFAULT_DIRECTORY):
    config = jupytext.config.JupytextConfiguration(root_level_metadata_as_raw_cell=False)
    for subdir in sorted(os.listdir(directory)):
        if not os.path.isdir(os.path.join(directory, subdir)):
            continue
        for filename in sorted(os.listdir(os.path.join(directory, subdir))):
            filename_abs = os.path.join(directory, subdir, filename)
            filename_base, ext = os.path.splitext(filename)
            if ext == ".py":
                module = f"examples.{subdir}.{filename_base}"
                data = jupytext.read(open(filename_abs), config=config)
                metadata = data["metadata"]["jupytext"].get("root_level_metadata", {})
                yield Example(ExampleType.MODULE, filename_abs, module, metadata)
            elif ext in [".png", ".jpeg", ".jpg", ".gif"]:
                yield Example(ExampleType.ASSET, filename_abs, None, None)


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

    example_name = filename.split("/")[-1]
    example_path = f"/api/raw-examples/{example_name}"
    markdown.append(
        f"\n_The raw source code for this example can be found [here]({example_path})._\n",
    )

    text = "\n".join(markdown)
    if _RE_FRONTMATTER.match(text):
        # Strip out frontmatter from text.
        if match := _RE_FRONTMATTER.search(text, 4):
            text = text[match.end() :]
    return text


if __name__ == "__main__":
    for example in get_examples():
        print(example)
