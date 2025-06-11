import json
import re
import warnings
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional
import requests

from pydantic import BaseModel

EXAMPLES_ROOT = Path(__file__).parent.parent

with warnings.catch_warnings():
    # This triggers some dumb warning in jupyter_core
    warnings.simplefilter("ignore")
    import jupytext
    import jupytext.config


class ExampleType(int, Enum):
    MODULE = 1
    ASSET = 2


class Example(BaseModel):
    type: ExampleType
    filename: str  # absolute filepath to example file
    module: Optional[str] = (
        None  # python import path, or none if file is not a py module.
    )
    # TODO(erikbern): don't think the module is used (by docs or monitors)?
    metadata: Optional[dict] = None
    repo_filename: str  # git repo relative filepath
    cli_args: Optional[list] = None  # Full command line args to run it
    stem: Optional[str] = None  # stem of path
    tags: Optional[list[str]] = None  # metadata tags for the example
    env: Optional[dict[str, str]] = None  # environment variables for the example


_RE_NEWLINE = re.compile(r"\r?\n")
_RE_FRONTMATTER = re.compile(r"^---$", re.MULTILINE)
_RE_CODEBLOCK = re.compile(r"\s*```[^`]+```\s*", re.MULTILINE)


def render_example_md(example: Example) -> str:
    """Render a Python code example to Markdown documentation format."""

    with open(example.filename) as f:
        content = f.read()

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
            if markdown and markdown[-1]:
                markdown.append("")
            if code or line:
                code.append(line)

    if code:
        markdown.extend(["```python", *code, "```", ""])

    text = "\n".join(markdown)
    if _RE_FRONTMATTER.match(text):
        # Strip out frontmatter from text.
        if match := _RE_FRONTMATTER.search(text, 4):
            text = text[match.end() + 1 :]

    if match := _RE_CODEBLOCK.match(text):
        filename = Path(example.filename).name
        if match.end() == len(text):
            # Special case: The entire page is a single big code block.
            text = f"""# Example ({filename})

This is the source code for **{example.module}**.
{text}"""

    return text


def gather_example_files(
    parents: list[str], subdir: Path, ignored: list[str], recurse: bool
) -> Iterator[Example]:
    config = jupytext.config.JupytextConfiguration(
        root_level_metadata_as_raw_cell=False
    )

    for filename in sorted(list(subdir.iterdir())):
        if filename.is_dir() and recurse:
            # Gather two-subdirectories deep, but no further.
            yield from gather_example_files(
                parents + [str(subdir.stem)], filename, ignored, recurse=False
            )
        else:
            filename_abs: str = str(filename.resolve())
            ext: str = filename.suffix
            if parents:
                repo_filename: str = (
                    f"{'/'.join(parents)}/{subdir.name}/{filename.name}"
                )
            else:
                repo_filename: str = f"{subdir.name}/{filename.name}"

            if ext == ".py" and filename.stem != "__init__":
                if parents:
                    parent_mods = ".".join(parents)
                    module = f"{parent_mods}.{subdir.stem}.{filename.stem}"
                else:
                    module = f"{subdir.stem}.{filename.stem}"
                data = jupytext.read(open(filename_abs), config=config)
                metadata = data["metadata"]["jupytext"].get("root_level_metadata", {})
                cmd = metadata.get("cmd", ["modal", "run", repo_filename])
                args = metadata.get("args", [])
                tags = metadata.get("tags", [])
                env = metadata.get("env", dict())
                yield Example(
                    type=ExampleType.MODULE,
                    filename=filename_abs,
                    metadata=metadata,
                    module=module,
                    repo_filename=repo_filename,
                    cli_args=(cmd + args),
                    stem=Path(filename_abs).stem,
                    tags=tags,
                    env=env,
                )
            elif ext in [".png", ".jpeg", ".jpg", ".gif", ".mp4"]:
                yield Example(
                    type=ExampleType.ASSET,
                    filename=filename_abs,
                    repo_filename=repo_filename,
                )
            else:
                ignored.append(str(filename))


def get_examples() -> Iterator[Example]:
    """Yield all Python module files and asset files relevant to building modal.com/docs."""
    if not EXAMPLES_ROOT.exists():
        raise Exception(
            f"Can't find directory {EXAMPLES_ROOT}. You might need to clone the modal-examples repo there."
        )

    ignored = []
    for subdir in sorted(
        p
        for p in EXAMPLES_ROOT.iterdir()
        if p.is_dir()
        and not p.name.startswith(".")
        and not p.name.startswith("internal")
        and not p.name.startswith("misc")
    ):
        yield from gather_example_files(
            parents=[], subdir=subdir, ignored=ignored, recurse=True
        )


def get_examples_json():
    examples = list(ex.dict() for ex in get_examples())
    return json.dumps(examples)


def _extract_links_from_file(file_path):
    """Extract all markdown-style links from a Python file's comments."""
    link_pattern = re.compile(r"\[.*?\]\((https?://[^\s)]+)\)")
    links = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("#"):
                matches = link_pattern.findall(line)
                links.extend(matches)

    return links


def _check_link(url):
    """Return True if the URL is valid (status < 400), else False."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        return response.status_code < 400
    except requests.RequestException:
        return False


def check_links(file_path):
    links = _extract_links_from_file(file_path)
    if not links:
        print(f"No links found in {file_path}")
        return True

    print(f"Checking {len(links)} links in {file_path}...")
    all_valid = True
    for url in links:
        valid = _check_link(url)
        print(f"{'✅' if valid else '❌'} {url}")
        if not valid:
            all_valid = False

    return all_valid


if __name__ == "__main__":
    for example in get_examples():
        print(example.model_dump_json())
