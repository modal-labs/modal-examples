import asyncio
import json
import sys
import threading
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
from board_store import Board, dispatch
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from server import create_server


@pytest.fixture
def board(tmp_path):
    return Board(tmp_path / "board.sqlite3")


def test_retry_and_persistence(board):
    args = dict(
        agent="connor-codex",
        message_id="hello",
        title="Approach",
        body="Review?",
        recipient="rahul-devin",
    )
    first = board.post(**args)
    assert Board(board.path).post(**args) == first
    assert board.list_threads()["threads"][0]["message_count"] == 1
    with pytest.raises(ValueError, match="different message"):
        board.post(**(args | {"body": "Changed"}))
    with pytest.raises(ValueError, match="different message"):
        board.post(**(args | {"title": "Changed"}))


def test_inbox_routing_and_receipts(board):
    root = board.post("connor", "root", "Hello everyone", title="Board")
    direct = board.post(
        "connor", "direct", "For Rahul", thread_id=root["thread_id"], recipient="rahul"
    )
    assert board.inbox("connor")["messages"] == []
    assert [m["id"] for m in board.inbox("rahul")["messages"]] == ["root", "direct"]
    assert [m["id"] for m in board.inbox("another")["messages"]] == ["root"]
    board.acknowledge("rahul", ["root", "direct"])
    board.acknowledge("rahul", ["root", "direct"])
    assert Board(board.path).inbox("rahul")["messages"] == []
    assert len(board.inbox("another")["messages"]) == 1
    assert len(board.read_thread(direct["thread_id"])["messages"]) == 2


def test_acknowledgement_is_atomic(board):
    board.post("connor", "root", "Broadcast", title="Board")
    board.post("connor", "private", "Other inbox", title="Other", recipient="another")
    with pytest.raises(ValueError):
        board.acknowledge("rahul", ["root", "private"])
    assert [m["id"] for m in board.inbox("rahul")["messages"]] == ["root"]


def test_pagination(board):
    board.post("connor", "root", "One", title="First")
    board.post("connor", "other", "Two", title="Second")
    board.post("rahul", "reply", "Three", thread_id="root")
    first = board.read_thread("root", limit=1)
    second = board.read_thread("root", after=first["next_after"], limit=1)
    assert first["has_more"] and not second["has_more"]
    assert [m["id"] for m in first["messages"] + second["messages"]] == [
        "root",
        "reply",
    ]
    threads = board.list_threads(limit=1)
    assert threads["has_more"]
    assert (
        board.list_threads(after=threads["next_after"])["threads"][0]["id"] == "other"
    )
    inbox = board.inbox("third", limit=2)
    assert inbox["has_more"]
    assert (
        board.inbox("third", after=inbox["next_after"])["messages"][0]["id"] == "reply"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"body": " "},
        {"body": "x" * 16001},
        {"title": None},
        {"title": ""},
        {"recipient": "bad/name"},
        {"thread_id": ""},
        {"message_id": "bad/name"},
    ],
)
def test_invalid_post(board, payload):
    args = dict(agent="connor", message_id="new", title="Title", body="hello")
    with pytest.raises(ValueError):
        board.post(**(args | payload))
    assert board.list_threads()["threads"] == []


@pytest.mark.parametrize(
    "payload", [{"after": -1}, {"limit": 0}, {"limit": 101}, {"limit": True}]
)
def test_invalid_page(board, payload):
    with pytest.raises(ValueError):
        board.inbox("rahul", **payload)


@pytest.fixture
def backend(tmp_path):
    """The production boundary is one serial dispatcher with a persistent database."""
    lock = threading.Lock()
    path = tmp_path / "board.sqlite3"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                with lock:
                    result = dispatch(path, **body)
                self.send_response(200)
            except (ValueError, TypeError) as exc:
                result = {"error": str(exc)}
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, *_args):
            pass

    service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{service.server_port}"
    finally:
        service.shutdown()
        service.server_close()
        thread.join()


@asynccontextmanager
async def client(url: str, agent: str):
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).resolve()), "--adapter", url, agent],
    )
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


async def invoke(session: ClientSession, tool: str, **args) -> dict[str, Any]:
    result = await session.call_tool(tool, args)
    assert not result.isError, result
    assert result.structuredContent is not None
    return result.structuredContent


def test_two_independent_mcp_clients(backend):
    async def conversation():
        async with client(backend, "connor-codex") as codex:
            names = {tool.name for tool in (await codex.list_tools()).tools}
            assert names == {
                "post_message",
                "list_threads",
                "read_thread",
                "check_inbox",
                "acknowledge_messages",
            }
            root = await invoke(
                codex,
                "post_message",
                body="Review my approach?",
                title="Connor + Rahul",
                recipient="rahul-devin",
                message_id="hello",
            )
            async with client(backend, "rahul-devin") as devin:
                inbox = await invoke(devin, "check_inbox")
                assert inbox["messages"] == [root]
                assert (await invoke(devin, "check_inbox"))["messages"] == [root]
                reply = await invoke(
                    devin,
                    "post_message",
                    body="Looks good",
                    thread_id=root["thread_id"],
                    recipient="connor-codex",
                    message_id="reply",
                )
                await invoke(devin, "acknowledge_messages", message_ids=[root["id"]])
                assert (await invoke(codex, "check_inbox"))["messages"] == [reply]
                assert (
                    await invoke(codex, "read_thread", thread_id=root["thread_id"])
                )["messages"] == [root, reply]
                assert (
                    await invoke(
                        codex,
                        "post_message",
                        body="Review my approach?",
                        title="Connor + Rahul",
                        recipient="rahul-devin",
                        message_id="hello",
                    )
                    == root
                )
                error = await codex.call_tool(
                    "post_message",
                    {
                        "body": "Conflict",
                        "title": "Connor + Rahul",
                        "message_id": "hello",
                    },
                )
                assert error.isError
                posts = await asyncio.gather(
                    *[
                        invoke(
                            codex if i % 2 else devin,
                            "post_message",
                            body=f"Concurrent {i}",
                            thread_id=root["thread_id"],
                            message_id=f"concurrent-{i}",
                        )
                        for i in range(20)
                    ]
                )
                assert len({post["seq"] for post in posts}) == 20
                assert (
                    len(
                        (
                            await invoke(
                                codex, "read_thread", thread_id=root["thread_id"]
                            )
                        )["messages"]
                    )
                    == 22
                )
                inbox = await invoke(devin, "check_inbox")
                await invoke(
                    devin,
                    "acknowledge_messages",
                    message_ids=[m["id"] for m in inbox["messages"]],
                )
            # A fresh process with the same identity retains acknowledgement state.
            async with client(backend, "rahul-devin") as devin:
                assert (await invoke(devin, "check_inbox"))["messages"] == []
                waiting = asyncio.create_task(
                    invoke(devin, "check_inbox", wait_seconds=1)
                )
                await asyncio.sleep(0.1)
                late = await invoke(
                    codex,
                    "post_message",
                    body="One more thing",
                    thread_id=root["thread_id"],
                    recipient="rahul-devin",
                    message_id="late",
                )
                assert (await waiting)["messages"] == [late]

    asyncio.run(conversation())


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--adapter":
        raise SystemExit("Test adapter only")

    async def call(action: str, agent: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient() as http:
            response = await http.post(
                sys.argv[2], json=dict(action=action, agent=agent, payload=payload)
            )
            result = response.json()
            if response.is_error:
                raise ValueError(result["error"])
            return result

    create_server(sys.argv[3], call).run(transport="stdio")
