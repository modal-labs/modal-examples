import json
import subprocess

import generate_diff_matrix as gdm
import pytest


def test_determine_diff_range_push():
    event = {"before": "commit1", "after": "commit2"}
    base, head = gdm.determine_diff_range(event, "push")
    assert base == "commit1"
    assert head == "commit2"


def test_determine_diff_range_pull():
    event = {
        "pull_request": {
            "base": {"sha": "base_sha"},
            "head": {"sha": "head_sha"},
        }
    }
    base, head = gdm.determine_diff_range(event, "pull_request")
    assert base == "base_sha"
    assert head == "head_sha"


def test_determine_diff_range_invalid_event():
    event = {}
    with pytest.raises(SystemExit):
        gdm.determine_diff_range(event, "unsupported_event")


def test_filter_files():
    files = [
        "example.py",
        "internal/test.py",
        "misc/skip.py",
        "script.js",
        "dir/another.py",
    ]
    filtered = gdm.filter_files(files)
    assert filtered == ["example.py", "dir/another.py"]


def test_get_changed_files(monkeypatch):
    class DummyCompletedProcess:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(args, capture_output, text, check):
        return DummyCompletedProcess("file1.py\nfile2.py\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    files = gdm.get_changed_files("base", "head")
    assert files == ["file1.py", "file2.py"]


def test_write_output(tmp_path, monkeypatch):
    temp_output = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(temp_output))

    gdm.write_output("test_key", "test_value")

    with open(temp_output, "r") as f:
        content = f.read()
    assert "test_key=test_value" in content


def test_main_push(monkeypatch, tmp_path):
    # simulate a push event by creating a temporary event JSON
    event_data = {"before": "commit1", "after": "commit2"}
    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps(event_data))

    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")

    output_file = tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    # override get_changed_files to simulate a git diff call
    def fake_get_changed_files(base, head):
        return ["file1.py", "internal/ignore.py", "misc/skip.py", "dir/keep.py"]

    monkeypatch.setattr(gdm, "get_changed_files", fake_get_changed_files)

    gdm.main()

    with open(output_file, "r") as f:
        output_content = f.read().strip()
    expected = json.dumps(["file1.py", "dir/keep.py"])
    assert f"all_changed_files={expected}" in output_content


def test_main_pull(monkeypatch, tmp_path):
    # simulate a pull_request event
    event_data = {
        "pull_request": {
            "base": {"sha": "base_commit"},
            "head": {"sha": "head_commit"},
        }
    }
    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps(event_data))

    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

    output_file = tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    def fake_get_changed_files(base, head):
        return [
            "pull_file.py",
            "internal/not_this.py",
            "misc/also_not.py",
            "folder/keep_this.py",
        ]

    monkeypatch.setattr(gdm, "get_changed_files", fake_get_changed_files)

    gdm.main()

    with open(output_file, "r") as f:
        output_content = f.read().strip()
    expected = json.dumps(["pull_file.py", "folder/keep_this.py"])
    assert f"all_changed_files={expected}" in output_content
