import importlib
import json
import pathlib
import sys

import pytest
from utils import (
    EXAMPLES_ROOT,
    ExampleType,
    get_examples,
    get_examples_json,
    render_example_md,
)

examples = [ex for ex in get_examples() if ex.type == ExampleType.MODULE]
examples = [ex for ex in examples if ex.metadata.get("pytest", True)]
example_ids = [ex.module for ex in examples]


@pytest.fixture(autouse=False)
def add_root_to_syspath(monkeypatch):
    sys.path.append(str(EXAMPLES_ROOT))
    yield
    sys.path.pop()


@pytest.mark.parametrize("example", examples, ids=example_ids)
def test_filename(example):
    assert not example.repo_filename.startswith("/")
    assert pathlib.Path(example.repo_filename).exists()


@pytest.mark.parametrize("example", examples, ids=example_ids)
def test_import(example, add_root_to_syspath):
    importlib.import_module(example.module)


@pytest.mark.parametrize("example", examples, ids=example_ids)
def test_render(example):
    md = render_example_md(example)
    assert isinstance(md, str)
    assert len(md) > 0


def test_json():
    data = get_examples_json()
    examples = json.loads(data)
    assert isinstance(examples, list)
    assert len(examples) > 0
