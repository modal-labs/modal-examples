"""This script heuristically adds force_build=True to the calls of modal.Image methods
in Python source files by searching and modifying the AST.

The heuristic fails with a false negative if the method call is on a variable
whose name either is not `image` or doesn't end with `_image`.

The heuristic fails with a false positive if a method with the same name as
one of the targeted methods of modal.Image is called on a variable whose name
either is `image` or ends with `_image`.

It can be tested with `pytest --doctest-modules internal/add_force_build.py`."""

import ast
import logging
import os
import sys
from pathlib import Path
from typing import Optional

log_level = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level), format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


TARGET_METHODS = {
    "apt_install",
    "dockerfile_commands",
    "from_aws_ecr",
    "from_dockerfile",
    "from_gcp_artifact_registry",
    "from_registry",
    "micromamba_install",
    "pip_install",
    "pip_install_private_repos",
    "pip_install_from_requirements",
    "pip_install_from_pyproject",
    "poetry_install_from_file",
    "run_commands",
    "run_function",
}

force_build = ast.keyword(arg="force_build", value=ast.Constant(value=True))


class AddForceBuild(ast.NodeTransformer):
    """Adds force_build=True to invocations of TARGET_METHODS on modal.Images.

    >>> source = '''
    ... import modal
    ... image = modal.Image().pip_install("pandas")
    ... '''
    >>> tree = ast.parse(source)
    >>> transformer = AddForceBuild()
    >>> new_source = ast.unparse(transformer.visit(tree))
    >>> "force_build=True" in new_source
    True
    """

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if method_name in TARGET_METHODS:
                obj = node.func.value
                if is_modal_image_chain(obj) or is_image_variable(obj):
                    logger.debug(f"Found target call: {ast.unparse(node)}")
                    if not has_force_build_already(node.keywords):
                        node.keywords.append(force_build)
        return self.generic_visit(node)


def is_image_variable(node):
    """Heuristic for checking whether a name refers to a Modal Image.

    >>> node = ast.Name(id='image', ctx=ast.Load())
    >>> is_image_variable(node)
    True
    >>> node = ast.Name(id='my_image', ctx=ast.Load())
    >>> is_image_variable(node)
    True
    >>> node = ast.Name(id='not_imag', ctx=ast.Load())
    >>> is_image_variable(node)
    False
    """
    if isinstance(node, ast.Name):
        return node.id == "image" or node.id.endswith("_image")
    return False


def has_force_build_already(kwargs):
    """Check if force_build is already in the keyword arguments."""
    return any(kw.arg == "force_build" for kw in kwargs)


def is_modal_image_chain(node):
    """Boolean indicating whether a chain of attributes starts with modal.Image

    >>> node = ast.parse('modal.Image').body[0].value
    >>> is_modal_image_chain(node)
    True
    >>> node = ast.parse('modal.Volume').body[0].value
    >>> is_modal_image_chain(node)
    False
    >>> node = ast.parse('PIL.Image.open').body[0].value
    >>> is_modal_image_chain(node)
    False
    """
    chain = extract_attr_chain(node)
    return chain[:2] == ["modal", "Image"]


def extract_attr_chain(node):
    """Parse a chain of attr accesses into a list of strings.

    >>> node = ast.parse('modal.Image().pip_install()').body[0].value
    >>> extract_attr_chain(node)[:2]
    ['modal', 'Image']
    >>> node = ast.parse('a.b.c').body[0].value
    >>> extract_attr_chain(node)
    ['a', 'b', 'c']
    """
    if isinstance(node, ast.Call):
        return extract_attr_chain(node.func)
    elif isinstance(node, ast.Attribute):
        return extract_attr_chain(node.value) + [node.attr]
    elif isinstance(node, ast.Name):
        return [node.id]
    else:
        return []


def main(files: list[str]) -> int:
    """Adds force_build=True to each file in a list of files.

    Args:
        files: Paths to input Python files

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    for file in files:
        try:
            source = read_source(file)
            new_source = transform_source(source)
            write_source(new_source, file)
        except Exception as e:
            logger.error(f"Failed to process {file}: {e}")
            return 1
    return 0


def read_source(input_file: str) -> str:
    try:
        return Path(input_file).read_text()
    except Exception as e:
        logger.error(f"Error reading {input_file}: {e}")
        raise


def split_frontmatter(source: str) -> tuple[str | None, str]:
    """Split source into frontmatter and code if frontmatter exists.

    Args:
        source: Full source text

    Returns:
        tuple of (frontmatter or None, code)

    >>> text = '''# ---
    ... # cmd: ["modal", "serve", "misc/ice_cream.py"]
    ... # ---
    ... print("# ---")'''
    >>> fm, code = split_frontmatter(text)
    >>> '["modal", "serve", "misc/ice_cream.py"]' in fm
    True
    >>> print(code)
    print("# ---")

    >>> text = '''print("no frontmatter")'''
    >>> fm, code = split_frontmatter(text)
    >>> print(fm is None)
    True
    >>> print(code)
    print("no frontmatter")
    """
    if source.startswith("# ---\n"):
        parts = source.split("# ---\n", 2)
        if len(parts) >= 3:
            frontmatter = f"# ---\n{parts[1]}# ---\n"
            code = "# ---\n".join(parts[2:])
            return frontmatter, code
    return None, source


def join_frontmatter(frontmatter: str | None, code: str) -> str:
    """Join frontmatter and code back together.

    Args:
        frontmatter: Frontmatter text or None
        code: Python code

    Returns:
        Combined source text

    >>> fm = '''# ---
    ... # key: value
    ... # ---
    ... '''
    >>> code = 'print("hello")'
    >>> print(join_frontmatter(fm, code))
    # ---
    # key: value
    # ---
    print("hello")

    >>> print(join_frontmatter(None, code))
    print("hello")
    """
    if frontmatter is None:
        return code
    return f"{frontmatter}{code}"


def transform_source(source: str) -> str:
    """Transform source code by adding force_build=True to modal.Image methods.

    Args:
        source: Python source code

    Returns:
        str: Transformed source code

    >>> code = '''
    ... import modal
    ... image = modal.Image().pip_install("pandas")
    ... '''
    >>> "force_build=True" in transform_source(code)
    True
    >>> code = '''# ---
    ... # lambda-test: false
    ... # ---
    ... import modal
    ... image = modal.Image().pip_install("pandas", force_build=False)
    ... '''
    >>> "force_build=True" not in transform_source(code)
    True
    >>> "lambda-test: false" in transform_source(code)
    True
    """
    fm, code = split_frontmatter(source)
    tree = ast.parse(code)
    transformer = AddForceBuild()
    new_tree = transformer.visit(tree)
    return join_frontmatter(fm, ast.unparse(new_tree))


def write_source(source: str, output_path: str) -> None:
    try:
        Path(output_path).write_text(source)
        logger.info(f"Successfully wrote output to {output_path}")
    except Exception as e:
        logger.error(f"Error writing to {output_path}: {e}")
        raise


if __name__ == "__main__":
    assert len(sys.argv) > 1, (
        "USAGE: add_force_build.py some_file.py path/to/other_file.py ..."
    )
    sys.exit(main(sys.argv[1:]))
