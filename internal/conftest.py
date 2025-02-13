import pytest


@pytest.fixture(autouse=True)
def disable_auto_mount(monkeypatch):
    monkeypatch.setenv("MODAL_AUTOMOUNT", "0")
    yield
