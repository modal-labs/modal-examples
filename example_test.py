import importlib
import pathlib
import pytest

from example_utils import ExampleType, get_examples, render_example_md

examples = [ex for ex in get_examples() if ex.type == ExampleType.MODULE]
example_ids = [ex.module for ex in examples]


@pytest.mark.parametrize("example", examples, ids=example_ids)
def test_filename(example):
    assert not example.repo_filename.startswith("/")
    assert pathlib.Path(example.repo_filename).exists()


@pytest.mark.parametrize("example", examples, ids=example_ids)
def test_import(example):
    importlib.import_module(example.module)


@pytest.mark.parametrize("example", examples, ids=example_ids)
def test_render(example):
    md = render_example_md(example)
    assert isinstance(md, str)
    assert len(md) > 0
