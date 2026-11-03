"""Private Modal backend for the agent message board; no HTTP endpoint."""

import os
from pathlib import Path

import modal

app = modal.App(os.environ.get("AGENT_BOARD_APP", "agent-message-board"))
volume = modal.Volume.from_name(
    os.environ.get("AGENT_BOARD_VOLUME", "agent-message-board-data"),
    create_if_missing=True,
    version=2,
)
image = modal.Image.debian_slim(python_version="3.12").add_local_file(
    Path(__file__).with_name("board_store.py"), "/root/board_store.py"
)


@app.function(
    image=image,
    volumes={"/data": volume},
    max_containers=1,
    timeout=60,
)
def request(action: str, agent: str, payload: dict) -> dict:
    from board_store import dispatch

    # All actions share one serial writer. Never add @modal.concurrent here.
    volume.reload()
    result = dispatch(Path("/data/board.sqlite3"), action, agent, payload)
    volume.commit()
    return result
