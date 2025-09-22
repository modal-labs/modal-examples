import contextlib
import os
from typing import List, Optional

import dotenv
from mcp.server.fastmcp import FastMCP
from repl import CommandResponse, Repl

dotenv.load_dotenv()


"""
This file specifies the MCP server and its tool with the Claude Desktop app.
It is automatically ran when the Claude Desktop app is open provided one appropriately
configures their `claude_desktop_config.json` file as specified in the README.
"""

sessionRepl: Optional[Repl] = None
snapshot_id_store_file = os.path.expanduser(os.getenv("SNAPSHOT_ID_FILE_PATH"))


mcp = FastMCP("modalrepl")

# This tool creates a new repl with the specified timeout and packages.
@mcp.tool()
async def create_repl(timeout: int = 600, packages: List[str] = []) -> None:
    # default timeout is 10 minute
    try:
        packages.extend(["fastapi", "uvicorn", "pydantic"])
        with (
            contextlib.redirect_stdout(open(os.devnull, "w")),
            contextlib.redirect_stderr(open(os.devnull, "w")),
        ):
            repl = await Repl.create(packages=packages, timeout=timeout)
        global sessionRepl
        sessionRepl = repl
    except Exception as exc:
        raise RuntimeError(f"Error creating REPL. {exc}")

# This tool executes a command in the current repl.
@mcp.tool()
async def exec_cmd(command: str) -> CommandResponse:
    try:
        if sessionRepl is None:
            raise RuntimeError("REPL not created")
        commands = Repl.parse_command(command)
        res = await sessionRepl.run(commands)
        return res
    except Exception as exc:
        raise RuntimeError(f"Error executing command: {exc}")

# This tool restores a repl from a snapshot.
@mcp.tool()
async def get_repl_from_snapshot() -> None:
    try:
        with open(snapshot_id_store_file, "r") as f:
            snapshot_id = f.read()
        repl = await Repl.from_snapshot(snapshot_id)
        global sessionRepl
        sessionRepl = repl
    except Exception as exc:
        raise RuntimeError(f"Error getting REPL from snapshot: {exc}")


@mcp.tool()  # This tool saves the snapshot id to a file
def end_repl_and_save_snapshot():
    try:
        if sessionRepl:
            snapshot_id = sessionRepl.kill()
            with open(snapshot_id_store_file, "w") as f:
                f.write(snapshot_id)
    except Exception as exc:
        raise RuntimeError(f"Error shutting down REPL: {exc}")


if __name__ == "__main__":
    mcp.run()
