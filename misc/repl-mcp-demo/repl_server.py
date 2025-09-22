# Copyright Modal Labs 2025
import io
from contextlib import redirect_stdout
from typing import Any, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

"""
This file specifies the HTTP server that is running inside of the sandbox.
It is used to execute commands in the repl.

"""

app = FastAPI(title="REPL Server")


_exec_context: Dict[str, Any] = {"__builtins__": __builtins__}


class ReplCommand(BaseModel):
    code: List[Tuple[str, Literal["exec", "eval"]]]


class ReplCommandResponse(BaseModel):
    result: str
    stdout: str

@app.post("/", status_code=status.HTTP_200_OK)
def run_exec(body: ReplCommand) -> ReplCommandResponse:
    commands = body.code
    stdout_redir_buffer = io.StringIO()
    try:
        for command in commands:
            if command[1] == "exec":
                with redirect_stdout(stdout_redir_buffer): # redirects stdout to a string buffer
                    exec(command[0], _exec_context, _exec_context)
            else:
                with redirect_stdout(stdout_redir_buffer):
                    res = eval(command[0], _exec_context, _exec_context)
                stdout = stdout_redir_buffer.getvalue()
                return ReplCommandResponse(result=str(res), stdout=stdout)
        return ReplCommandResponse(
            result="", stdout=stdout_redir_buffer.getvalue()
        )  # just send back blank str if all commands are exec'd
    except Exception as exc:
        raise HTTPException(status_code=500, detail=repr(exc))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
