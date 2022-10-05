import importlib

from example_utils import ExampleType, get_examples, render_example_md


def test_examples():
    for example in get_examples():
        if example.type == ExampleType.MODULE:
            # If it's a module, try to import it
            importlib.import_module(example.module)

            # Let's also try to turn it in to Markdown
            md = render_example_md(example)
            assert isinstance(md, str)
            assert len(md) > 0
