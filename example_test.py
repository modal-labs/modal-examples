import importlib

from example_utils import ExampleType, get_examples

def test_examples():
    for example in get_examples():
        if example.type == ExampleType.MODULE:
            importlib.import_module(example.module)
