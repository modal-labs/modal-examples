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


def main(
    input_file: str, in_place: bool = False, output_file: Optional[str] = None
) -> int:
    """Main function to process Python files and add force_build=True.

    Args:
        input_file: Path to input Python file
        in_place: Whether to modify the file in place
        output_file: Path to output file (if not in_place)

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        source = read_source(input_file)
        new_source = transform_source(source)
        out_path = input_file if in_place else output_file
        write_source(new_source, out_path)
        return 0
    except Exception as e:
        logger.error(f"Failed to process {input_file}: {e}")
        return 1


def get_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Add force_build=True to modal.Image method calls."
    )
    parser.add_argument("input_file", help="Path to the input Python file")
    parser.add_argument("--output_file", help="Path for the output Python file")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify the file in place instead of writing to a separate output file",
    )
    return parser.parse_args()


def read_source(input_file: str) -> str:
    """Read source code from a file.

    Args:
        input_file: Path to the input file

    Returns:
        str: Source code content

    Raises:
        FileNotFoundError: If the input file doesn't exist
    """
    try:
        return Path(input_file).read_text()
    except Exception as e:
        logger.error(f"Error reading {input_file}: {e}")
        raise


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

    >>> code = '''
    ... import modal
    ... image = modal.Image().pip_install("pandas", force_build=False)
    ... '''
    >>> "force_build=True" not in transform_source(code)
    True
    """
    tree = ast.parse(source)
    transformer = AddForceBuild()
    new_tree = transformer.visit(tree)
    return ast.unparse(new_tree)


def write_source(source: str, output_path: str) -> None:
    """Write transformed source to output file.

    Args:
        source: Transformed source code
        output_path: Path to write the output
    """
    try:
        Path(output_path).write_text(source)
        logger.info(f"Successfully wrote output to {output_path}")
    except Exception as e:
        logger.error(f"Error writing to {output_path}: {e}")
        raise


if __name__ == "__main__":
    args = get_args()
    sys.exit(main(**vars(args)))
