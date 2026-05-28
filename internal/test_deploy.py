import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import deploy as dep


class FakeCompletedProcess:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_deploy_uses_correct_cwd(monkeypatch):
    """deploy() must set cwd to the example file's parent directory, not '.'."""
    captured = {}

    def fake_run(args, cwd, capture_output, env):
        captured["cwd"] = cwd
        return FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    example_file = Path("/repo/06_gpu_and_ml/text_to_speech.py")
    dep.deploy(
        deployable=True,
        module_with_app=example_file,
        dry_run=False,
        filter_pttrn=None,
        env={},
    )

    assert captured["cwd"] == Path("/repo/06_gpu_and_ml"), (
        f"Expected cwd to be the example's directory, got {captured['cwd']!r}"
    )


def test_deploy_uses_stem_for_module_flag(monkeypatch):
    """deploy() must pass the file stem (no extension) to 'modal deploy -m'."""
    captured = {}

    def fake_run(args, cwd, capture_output, env):
        captured["args"] = args
        return FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    example_file = Path("/repo/06_gpu_and_ml/text_to_speech.py")
    dep.deploy(
        deployable=True,
        module_with_app=example_file,
        dry_run=False,
        filter_pttrn=None,
        env={},
    )

    assert "text_to_speech" in captured["args"], (
        "Expected module stem in deploy args"
    )
    assert "text_to_speech.py" not in captured["args"], (
        "File extension must not appear in deploy -m argument"
    )


def test_deploy_skips_non_deployable(monkeypatch):
    """deploy() returns None immediately when deployable=False."""
    called = []

    def fake_run(*args, **kwargs):
        called.append(True)
        return FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = dep.deploy(
        deployable=False,
        module_with_app=Path("/repo/06_gpu_and_ml/text_to_speech.py"),
        dry_run=False,
        filter_pttrn=None,
        env={},
    )

    assert result is None
    assert not called, "subprocess.run must not be called for non-deployable examples"
