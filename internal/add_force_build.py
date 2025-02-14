import sys
from pathlib import Path

import libcst as cst
import libcst.matchers as m
import modal

# These are the Image construction methods that we want to add force_build to.

# NOTE: these methods are in lexical order
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

image_methods = set(dir(modal.Image))
assert (image_methods & TARGET_METHODS) == TARGET_METHODS, (
    f"Unknown modal.Image method(s): {TARGET_METHODS - image_methods}"
)

# DO NOT add any of the methods that construct a base image,
# since this will trigger a rebuild of that image and all dependent images
base_image_methods = {"debian_slim", "micromamba"}
assert (base_image_methods & TARGET_METHODS) == set(), (
    f"Cannot force build base image method(s): {base_image_methods & TARGET_METHODS}"
)


class ForceBuildTransformer(cst.CSTTransformer):
    """Adds force_build to targeted Image method calls."""

    def __init__(self):
        self.image_vars = set()

    def is_modal_image_expr(self, expr: cst.BaseExpression) -> bool:
        """
        Recursively check whether `expr` ultimately originates from a modal.Image.
        """
        if isinstance(expr, cst.Call):
            return self.is_modal_image_expr(expr.func)

        if isinstance(expr, cst.Attribute):
            if m.matches(
                expr, m.Attribute(value=m.Name("modal"), attr=m.Name("Image"))
            ):
                return True
            return self.is_modal_image_expr(expr.value)

        if isinstance(expr, cst.Name):
            return expr.value in self.image_vars

        return False

    def leave_Assign(
        self, original_node: cst.Assign, updated_node: cst.Assign
    ) -> cst.Assign:
        """Track assignments of names to modal.Image objects."""
        if m.matches(
            original_node.value,
            m.Call(
                func=m.Attribute(
                    value=m.Attribute(
                        value=m.Name("modal"), attr=m.Name("Image")
                    ),
                    attr=m.Name(),
                )
            ),
        ):
            call_func = original_node.value.func
            if isinstance(call_func, cst.Attribute):
                method_name = call_func.attr.value
                if method_name in TARGET_METHODS:
                    for target in original_node.targets:
                        if m.matches(target.target, m.Name()):
                            var_name = target.target.value
                            self.image_vars.add(var_name)
        return updated_node

    def leave_Call(
        self, original_node: cst.Call, updated_node: cst.Call
    ) -> cst.Call:
        """
        Modify calls to targeted methods.
        """
        if not isinstance(original_node.func, cst.Attribute):
            return updated_node

        method_name = original_node.func.attr.value
        if method_name not in TARGET_METHODS:
            return updated_node

        if not self.is_modal_image_expr(original_node.func.value):
            return updated_node

        for arg in original_node.args:
            if arg.keyword is not None and arg.keyword.value == "force_build":
                return updated_node

        new_arg = cst.Arg(
            keyword=cst.Name("force_build"), value=cst.Name("True")
        )
        new_args = list(updated_node.args) + [new_arg]
        return updated_node.with_changes(args=new_args)


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        source = Path(filename).read_text()
    else:
        source = TEST_CASE

    module = cst.parse_module(source)
    transformer = ForceBuildTransformer()
    modified_tree = module.visit(transformer)
    transformed_code = modified_tree.code

    if len(sys.argv) > 1:
        Path(filename).write_text(transformed_code)
    else:
        assert (
            transformed_code.count("force_build = True") == TEST_CASE["count"]
        )


TEST_CASE = {
    "source": """import modal

img1 = modal.Image.debian_slim().pip_install("package==1.0")  # targeted, 1
img2 = modal.Image.debian_slim()  # not targeted

img3 = modal.Image.from_dockerfile("Dockerfile")  # targeted, 2
img3.run_commands("echo hello")  # targeted, 3

def inside_function(image):
    # This call will be missed because 'image' isn't tracked
    image.pip_install("other_package==2.0")
""",
    "count": 3,
}

if __name__ == "__main__":
    main()
