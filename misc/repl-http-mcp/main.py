from mcp.server.fastmcp import FastMCP
from typing import Optional, List
from repl import ReplMCPExecResponse
import dotenv
import os
import httpx
import asyncio

dotenv.load_dotenv()

sessionRepl: Optional[str] = None
snapshot_id_store_file = os.path.expanduser(os.getenv("SNAPSHOT_ID_FILE_PATH"))
server_url = os.getenv("HTTP_SERVER_URL")
request_timeout = int(os.getenv("REQUEST_TIMEOUT"))



mcp = FastMCP("modalrepl")


@mcp.tool()
async def create_repl(timeout: int = 600, packages: List[str] = []) -> None:
    # default timeout is 10 minute
    try:
        async with httpx.AsyncClient() as client:
            create_fut = client.post(f"{server_url}/create_repl",json={"timeout": timeout, "packages": packages})
            response = await asyncio.wait_for(create_fut, timeout=20)
            repl_id = response.json()["repl_id"]
            global sessionRepl
            sessionRepl = repl_id
    except Exception as exc:
        raise RuntimeError(f"HTTP error creating REPL. Your REPL may have timed out. {exc}")

@mcp.tool()
async def exec_cmd(command: str) -> ReplMCPExecResponse:
    try:
        if sessionRepl is None:
            raise RuntimeError("REPL not created")
        async with httpx.AsyncClient() as client:
            exec_fut = client.post(f"{server_url}/exec",json={"repl_id": sessionRepl, "command": command})
            response = await asyncio.wait_for(exec_fut, timeout=20)
        return response.json()
    except Exception as exc:
        raise RuntimeError(f"HTTP error executing command: {exc}")



if __name__ == "__main__":
    mcp.run()

