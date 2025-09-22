import ast
import os
import uuid
from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel
import httpx

from modal.app import App
from modal.image import Image
from modal.output import enable_output
from modal.sandbox import Sandbox
from modal.snapshot import SandboxSnapshot



class ReplMCPExecResponse(BaseModel):
    output: Optional[str]
    stdout: Optional[str]
    error: Optional[str]



class Repl:

    def __init__(self, sandbox: Sandbox, sb_url: str, id: Optional[str] = None):
        self.sb = sandbox
        self.sb_url = sb_url
        self.id = id or str(uuid.uuid4())

    @staticmethod
    def parse_command(code: str) -> List[Tuple[str, Literal["exec", "eval"]]]:
        try:
            tree = ast.parse(code, mode="exec")
            if tree.body and len(tree.body) > 0 and isinstance(tree.body[-1], ast.Expr):  # ast.Expr should be eval()'d
                last_expr = tree.body[-1]
                lines = code.splitlines(keepends=True)
                start_line = getattr(last_expr, "lineno", None)
                start_col = getattr(last_expr, "col_offset", None)
                end_line = getattr(last_expr, "end_lineno", None)
                end_col = getattr(last_expr, "end_col_offset", None)
                # print(start_line, start_col, end_line, end_col)
                if end_line is None or end_col is None or start_line is None or start_col is None:
                    return [(code, "exec")]
                start_line -= 1
                end_line -= 1  # ast parser returns 1-indexed lines.our list of strings is 0-indexed
                prefix_parts = []
                if start_line > 0:
                    prefix_parts.append("".join(lines[:start_line]))
                prefix_parts.append(lines[start_line][:start_col])
                prefix_code = "".join(prefix_parts)
                # puts everything before last expression into one str. this is all exec()'d
                last_expr_parts = []
                if start_line == end_line:
                    last_expr_parts.append(lines[start_line][start_col:end_col])
                else:
                    last_expr_parts.append(lines[start_line][start_col:])
                    if end_line - start_line > 1:
                        last_expr_parts.append("\n".join(lines[start_line + 1 : end_line]))
                    last_expr_parts.append(lines[end_line][:end_col])
                last_expr_code = "".join(last_expr_parts)

                commands = []
                if prefix_code.strip():
                    commands.append((prefix_code, "exec"))
                commands.append((last_expr_code, "eval"))
            else:
                commands = [(code, "exec")]  # whole thing exec()'d
            returnCommands = []
            for cmd in commands:
                if cmd[0].strip():
                    returnCommands.append(cmd)
            return returnCommands
        except Exception as e:
            print(repr(e))
            return []

    @staticmethod
    async def create(python_version: str = "3.13", port: int = 8000, packages: List[str] = [], timeout: int = 600) -> "Repl":
        try:
            image = Image.debian_slim(python_version=python_version)
            image = image.pip_install(*packages)
            repl_server_path = os.path.join(os.path.dirname(__file__), "repl_server.py")
            image = image.add_local_file(local_path=repl_server_path, remote_path="/root/repl_server.py")
            app = App.lookup(name="repl", create_if_missing=True)
            with enable_output():
                start_cmd = ["bash", "-c", "cd /root && python repl_server.py"]
                sb = await Sandbox.create.aio(
                    *start_cmd, app=app, image=image, encrypted_ports=[port], _experimental_enable_snapshot=True, timeout=timeout
                )
                sb_url = (await sb.tunnels.aio())[port].url
            return Repl(sb, sb_url)
        except Exception as e:
            raise Exception(f"{e}")

    @staticmethod
    async def from_snapshot(snapshot_id: str, id: Optional[str] = None) -> "Repl":
        try: 
            snapshot = await SandboxSnapshot.from_id.aio(snapshot_id)
            sb = await Sandbox._experimental_from_snapshot.aio(snapshot)
            sb_url = (await sb.tunnels.aio())[8000].url
            return Repl(sb, sb_url, id)
        except Exception as e:
            raise Exception(f"Error getting repl from snapshot: {repr(e)}")
            

    async def run(self, commands: List[Tuple[str, Literal["exec", "eval"]]]) -> ReplMCPExecResponse:
        try:
            async with httpx.AsyncClient() as client:
                repl_output = await client.post(self.sb_url, json={"code": commands})
                if repl_output.status_code != 200:
                    err = repl_output.json()["detail"]
                    return ReplMCPExecResponse(error=err, output=None, stdout=None)
                output = repl_output.json()["result"]
                stdout = repl_output.json()["stdout"]
                stdout_lines = stdout.splitlines()
                stdout_lines = [line for line in stdout_lines if not line.startswith("INFO:")] # bad sol to ignore uvicorn logs
                return ReplMCPExecResponse(output=output, stdout="\n".join(stdout_lines), error=None)
        except Exception as e:
            raise Exception(f"Error running commands: {repr(e)}")

    def kill(self) -> str:
        try: 
            if self.sb:
                snapshot = self.sb._experimental_snapshot
                self.sb.terminate()
                return snapshot.object_id
        except Exception as e:
            raise Exception(f"Error killing repl: {repr(e)}")