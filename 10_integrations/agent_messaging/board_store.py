"""Single-writer storage. Close every SQLite handle before committing the Volume."""

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[\w.:-]{1,128}", value):
        raise ValueError("Identifiers must be 1–128 letters, digits, _, ., :, or -")
    return value


def text(value: Any, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(
            f"Text must contain 1–{maximum} characters, not just whitespace"
        )
    return value


def page(after: int, limit: int) -> None:
    if type(after) is not int or after < 0:
        raise ValueError("after must be a nonnegative integer")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100")


class Board:
    def __init__(self, path: Path):
        self.path = path
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    thread_id TEXT NOT NULL REFERENCES threads(id),
                    sender TEXT NOT NULL,
                    recipient TEXT,
                    body TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_thread ON messages(thread_id, seq);
                CREATE TABLE IF NOT EXISTS receipts (
                    agent TEXT NOT NULL,
                    message_id TEXT NOT NULL REFERENCES messages(id),
                    PRIMARY KEY (agent, message_id)
                );
            """)

    def connect(self):
        # sqlite3's transaction context manager does not close the connection.
        from contextlib import closing

        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return closing(db)

    def post(
        self,
        agent: str,
        message_id: str,
        body: str,
        thread_id: str | None = None,
        title: str | None = None,
        recipient: str | None = None,
    ) -> dict[str, Any]:
        identifier(agent)
        identifier(message_id)
        text(body, 16000)
        if recipient is not None:
            identifier(recipient)
        new_thread = thread_id is None
        thread_id = identifier(message_id if new_thread else thread_id)
        if new_thread:
            text(title, 200)
        elif title is not None:
            raise ValueError("Only new threads take a title")
        with self.connect() as db, db:
            existing = db.execute(
                "SELECT * FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
            if existing is not None:
                expected = (thread_id, agent, recipient, body)
                actual = tuple(
                    existing[k] for k in ("thread_id", "sender", "recipient", "body")
                )
                saved_title = db.execute(
                    "SELECT title FROM threads WHERE id = ?", (thread_id,)
                ).fetchone()
                if actual != expected or (new_thread and saved_title[0] != title):
                    raise ValueError(
                        "message_id was already used for a different message"
                    )
                return dict(existing)
            if new_thread:
                db.execute("INSERT INTO threads VALUES (?, ?)", (thread_id, title))
            elif not db.execute(
                "SELECT 1 FROM threads WHERE id = ?", (thread_id,)
            ).fetchone():
                raise ValueError("Thread does not exist")
            db.execute(
                """INSERT INTO messages (id, thread_id, sender, recipient, body, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    message_id,
                    thread_id,
                    agent,
                    recipient,
                    body,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return dict(
                db.execute(
                    "SELECT * FROM messages WHERE id = ?", (message_id,)
                ).fetchone()
            )

    def read_thread(
        self, thread_id: str, after: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        identifier(thread_id)
        page(after, limit)
        with self.connect() as db:
            thread = db.execute(
                "SELECT * FROM threads WHERE id = ?", (thread_id,)
            ).fetchone()
            if thread is None:
                raise ValueError("Thread does not exist")
            rows = db.execute(
                "SELECT * FROM messages WHERE thread_id = ? AND seq > ? ORDER BY seq LIMIT ?",
                (thread_id, after, limit + 1),
            ).fetchall()
            return dict(thread) | self.message_page(rows, after, limit)

    @staticmethod
    def message_page(rows, after: int, limit: int) -> dict[str, Any]:
        messages = [dict(row) for row in rows[:limit]]
        return {
            "messages": messages,
            "next_after": messages[-1]["seq"] if messages else after,
            "has_more": len(rows) > limit,
        }

    def list_threads(self, after: int = 0, limit: int = 50) -> dict[str, Any]:
        page(after, limit)
        with self.connect() as db:
            rows = db.execute(
                """SELECT t.id, t.title, MIN(m.seq) AS first_seq, MAX(m.seq) AS last_seq,
                          COUNT(*) AS message_count
                   FROM threads t JOIN messages m ON m.thread_id = t.id
                   GROUP BY t.id HAVING MIN(m.seq) > ?
                   ORDER BY first_seq LIMIT ?""",
                (after, limit + 1),
            ).fetchall()
            threads = [dict(row) for row in rows[:limit]]
            return {
                "threads": threads,
                "next_after": threads[-1]["first_seq"] if threads else after,
                "has_more": len(rows) > limit,
            }

    def inbox(self, agent: str, after: int = 0, limit: int = 50) -> dict[str, Any]:
        identifier(agent)
        page(after, limit)
        with self.connect() as db:
            rows = db.execute(
                """SELECT m.* FROM messages m
                   WHERE m.seq > ? AND m.sender != ?
                     AND (m.recipient IS NULL OR m.recipient = ?)
                     AND NOT EXISTS (SELECT 1 FROM receipts r WHERE r.agent = ? AND r.message_id = m.id)
                   ORDER BY m.seq LIMIT ?""",
                (after, agent, agent, agent, limit + 1),
            ).fetchall()
            return self.message_page(rows, after, limit)

    def acknowledge(self, agent: str, message_ids: list[str]) -> dict[str, int]:
        identifier(agent)
        if not isinstance(message_ids, list) or not 1 <= len(message_ids) <= 100:
            raise ValueError("Provide 1–100 message IDs")
        for message_id in message_ids:
            identifier(message_id)
        with self.connect() as db, db:
            for message_id in message_ids:
                row = db.execute(
                    "SELECT sender, recipient FROM messages WHERE id = ?", (message_id,)
                ).fetchone()
                if (
                    row is None
                    or row["sender"] == agent
                    or row["recipient"] not in (None, agent)
                ):
                    raise ValueError("Message is not addressed to this agent")
                db.execute(
                    "INSERT OR IGNORE INTO receipts VALUES (?, ?)", (agent, message_id)
                )
        return {"acknowledged": len(set(message_ids))}


def dispatch(path: Path, action: str, agent: str, payload: dict) -> dict:
    identifier(agent)
    board = Board(path)
    match action:
        case "post":
            return board.post(agent=agent, **payload)
        case "read_thread":
            return board.read_thread(**payload)
        case "list_threads":
            return board.list_threads(**payload)
        case "inbox":
            return board.inbox(agent=agent, **payload)
        case "acknowledge":
            return board.acknowledge(agent=agent, **payload)
        case _:
            raise ValueError("Unknown action")
