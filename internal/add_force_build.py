import ast
import sys
from pathlib import Path


def get_args():
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


def main(input_file, in_place=False, output_file=None):
    try:
        source = Path(input_file).read_text()
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        sys.exit(1)

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Syntax error in {input_file}: {e}")
        sys.exit(1)

    transformer = AddForceBuild()
    new_tree = transformer.visit(tree)

    try:
        new_source = ast.unparse(new_tree)
    except Exception as e:
        print(f"Error unparsing AST: {e}")
        sys.exit(1)

    out_path = input_file if in_place else output_file

    try:
        Path(out_path).write_text(new_source)
    except Exception as e:
        print(f"Error writing to {out_path}: {e}")
        sys.exit(1)

    print(f"Successfully processed {input_file} -> {out_path}")
    sys.exit(0)


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
    """Adds force_build=True to invocations of TARGET_METHODS on modal.Images."""

    def visit_Call(self, node):
        # if it's a method call
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            # that we want to add force_build=True to
            if method_name in TARGET_METHODS:
                obj = node.func.value
                # heuristically check if we're on a modal.Image
                if is_modal_image_chain(obj) or is_image_variable(obj):
                    print("Found target call:", ast.unparse(node))
                    # and add force_build if it's not there already
                    if not has_force_build_already(node.keywords):
                        node.keywords.append(force_build)
        return self.generic_visit(node)


def is_image_variable(node):
    """Heuristic for checking whether a name refers to a Modal Image.

    False-positives are OK here, since we will only attempt to modify
    the source if the method called on the variable is one of the TARGET_METHODS.
    """
    if isinstance(node, ast.Name):
        return node.id == "image" or node.id.endswith("_image")
    return False


def has_force_build_already(kwargs):
    return any(kw.arg == "force_build" for kw in kwargs)


def is_modal_image_chain(node):
    """Boolean indicating whether a chain of attributes starts with modal.Image"""
    chain = reduce_attr_chain(node)
    return chain[:2] == ["modal", "Image"]


def reduce_attr_chain(node):
    """Recursively reduce an attr chain down to a name or a name and attr.

    For example, for modal.Image(...).pip_install(...).env(...), this returns ['modal', 'Image'].
    """
    if isinstance(node, ast.Call):
        return reduce_attr_chain(node.func)
    elif isinstance(node, ast.Attribute):
        return reduce_attr_chain(node.value) + [node.attr]
    elif isinstance(node, ast.Name):
        return [node.id]
    else:
        return []


if __name__ == "__main__":
    main(**vars(get_args()))
