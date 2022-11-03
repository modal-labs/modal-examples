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
    repo_filename: str


_RE_NEWLINE = re.compile(r"\r?\n")
_RE_FRONTMATTER = re.compile(r"^---$", re.MULTILINE)


def render_example_md(example: Example) -> str:
    """Render a Python code example to Markdown documentation format."""

    with open(example.filename) as f:
        content = f.read()

    lines = _RE_NEWLINE.split(content)
    markdown: list[str] = []

    # temporary buffers for comment and code lines
    comments: list[str] = []
    code: list[str] = []

    for line in lines:
        if line == "#" or line.startswith("# "):
            if code:
                # write buffers into markdown
                markdown.extend(["", "<section class=\"md-code\">", "", "```python", *code, "```", "", "</section>"])
                if comments:
                    markdown.extend(["", "<section class=\"md-annotation\"><div>", "", *comments, "", "</div></section>"])
                else:
                    markdown.extend(["", "<section />"])
                comments = []
                code = []

            heading = line[2:].startswith("#")
            if heading:
                markdown.extend(["", "<section class=\"md-text\">", "", *comments, "", "</section>"])
                markdown.extend(["", "<section class=\"md-header\">", "", line[2:], "", "</section>"])
                comments = []
            else:
                comments.append(line[2:])

            if line[2:].startswith("-"):
                comments = []
        else:
            if code or line:
                code.append(line)

    if code and comments:
        markdown.extend(["", "<section class=\"md-code\">", "", "```python", *code, "```", "", "</section>"])
        markdown.extend(["", "<section class=\"md-annotation\"><div>", "", *comments, "", "</div></section>"])
        code = []

    github_url = f"https://github.com/modal-labs/modal-examples/blob/main/{example.repo_filename}"
    markdown.extend([
        "", "<section class=\"md-text\">", "",
        f"\n_The raw source code for this example can be found [on GitHub]({github_url})._\n",
        "", "</section>",
    ])

    text = "\n".join(markdown)
    if _RE_FRONTMATTER.match(text):
        # Strip out frontmatter from text.
        if match := _RE_FRONTMATTER.search(text, 4):
            text = text[match.end() :]
    return text


def get_examples(directory: Path = DEFAULT_DIRECTORY):
    if not directory.exists():
        raise Exception(f"Can't find directory {directory}. You might need to clone the modal-examples repo there")

    config = jupytext.config.JupytextConfiguration(root_level_metadata_as_raw_cell=False)
    ignored = []
    for subdir in sorted(list(directory.iterdir())):
        if not subdir.is_dir():
            continue
        for filename in sorted(list(subdir.iterdir())):
            filename_abs: str = str(filename.resolve())
            ext: str = filename.suffix
            repo_filename: str = f"{subdir.name}/{filename.name}"
            if ext == ".py":
                module = f"{subdir.stem}.{filename.stem}"
                data = jupytext.read(open(filename_abs), config=config)
                metadata = data["metadata"]["jupytext"].get("root_level_metadata", {})
                yield Example(
                    ExampleType.MODULE,
                    filename_abs,
                    module,
                    metadata,
                    repo_filename,
                )
            elif ext in [".png", ".jpeg", ".jpg", ".gif", ".mp4"]:
                yield Example(ExampleType.ASSET, filename_abs, None, None, repo_filename)
            else:
                ignored.append(str(filename))
    print(f"Ignoring examples files: {ignored}")


if __name__ == "__main__":
    for example in get_examples():
        print(example)
