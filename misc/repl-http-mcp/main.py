import asyncio
import os
from typing import List, Optional

import dotenv
import httpx
from mcp.server.fastmcp import FastMCP
from repl import ReplMCPExecResponse

dotenv.load_dotenv()

sessionRepl: Optional[str] = None
server_url = os.getenv("HTTP_SERVER_URL")
repl_id_file = os.getenv("REPL_ID_FILE")


mcp = FastMCP("modalrepl")


# repl creation. called upon first use of MCP.
async def create_repl(timeout: int = 30, packages: List[str] = []) -> None:
    # default timeout is 30s
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{server_url}/create_repl",
                json={"timeout": timeout, "packages": packages},
            )
            print(response.json())
            repl_id = response.json()["repl_id"]
            global sessionRepl
            sessionRepl = repl_id
    except Exception as exc:
        print(exc)
        raise RuntimeError(
            f"HTTP error creating REPL. Your REPL may have timed out. {exc}"
        )


# executes arbitrary code in repl. this is the only tool call accessible.
@mcp.tool()
async def exec_cmd(command: str) -> ReplMCPExecResponse:
    try:
        if sessionRepl is None:
            raise RuntimeError("REPL not created")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{server_url}/exec", json={"repl_id": sessionRepl, "command": command}
            )
        return response.json()
    except Exception as exc:
        raise RuntimeError(f"HTTP error executing command: {exc}")


# start_session will check the dotfile for a persisted repl id. if none, it will create a new repl.
def start_session() -> None:
    repl_id_file_path = os.path.expanduser(repl_id_file)
    with open(repl_id_file_path, "r") as f:  # check for persisted repl id
        repl_id = f.read()
    if repl_id:
        global sessionRepl
        sessionRepl = repl_id
    else:
        asyncio.run(create_repl())
        with open(repl_id_file_path, "w") as f:
            f.write(sessionRepl)


if __name__ == "__main__":
    start_session()
    mcp.run()
