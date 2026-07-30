import importlib
import json
import pathlib
import sys

import pytest
from utils import (
    EXAMPLES_ROOT,
    Example,
    ExampleType,
    get_examples,
    get_examples_json,
    render_example_md,
)

examples = [ex for ex in get_examples() if ex.type == ExampleType.MODULE]
examples = [ex for ex in examples if ex.metadata.get("pytest", True)]
example_ids = [ex.module for ex in examples]


@pytest.fixture(autouse=True)
def disable_auto_mount(monkeypatch):
    monkeypatch.setenv("MODAL_AUTOMOUNT", "0")
    yield


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


def render_source(tmp_path, source: str) -> str:
    path = tmp_path / "pep723_fixture.py"
    path.write_text(source)
    return render_example_md(
        Example(
            type=ExampleType.MODULE,
            filename=str(path),
            repo_filename="pep723_fixture.py",
            stem="pep723_fixture",
            module="pep723_fixture",
        )
    )


def test_render_fences_pep723(tmp_path):
    md = render_source(
        tmp_path,
        """# ---
# cmd: ["uv", "run", "--script", "pep723_fixture.py"]
# ---
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "harbor==0.15.0",
# ]
# ///

# # Fixture title for docs
#
# Some prose.

x = 1
""",
    )
    fenced = '```python\n# /// script\n# requires-python = ">=3.12"\n# dependencies = [\n#   "harbor==0.15.0",\n# ]\n# ///\n```'
    assert fenced in md
    assert not md.lstrip().startswith("/// script")
    assert md.index("```python") < md.index("# Fixture title for docs")
    assert "Fixture title for docs" in md


def test_render_fences_arbitrary_pep723_type(tmp_path):
    md = render_source(
        tmp_path,
        '''# /// some-toml
# embedded-csharp = """
# ///
# /// text
# ///
# public class MyClass { }
# """
# ///

x = 1
''',
    )
    fenced = '```python\n# /// some-toml\n# embedded-csharp = """\n# ///\n# /// text\n# ///\n# public class MyClass { }\n# """\n# ///\n```'
    assert fenced in md


def test_render_ignores_unterminated_pep723(tmp_path):
    md = render_source(
        tmp_path,
        """# /// script
# requires-python = ">=3.12"

x = 1
""",
    )
    assert "```python\n# /// script" not in md
    assert md.lstrip().startswith("/// script")
