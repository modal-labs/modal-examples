import asyncio
import logging
import os
from typing import List, Optional

import dotenv
import fastapi
import modal
from fastapi import HTTPException, status
from modal import App, Dict, Image, Secret
from pydantic import BaseModel

dotenv.load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




class ReplMCPCreateRequest(BaseModel):
    python_version: str = "3.13"
    packages: List[str] = []
    port: int = 8000
    timeout: float = 1 # num of minutes to keep repl alive


class ReplMCPCreateResponse(BaseModel):
    repl_id: str


class ReplMCPExecRequest(BaseModel):
    repl_id: str
    command: str


class ReplMCPExecResponse(BaseModel):
    output: Optional[str]
    stdout: Optional[str]
    error: Optional[str]

image = Image.debian_slim(python_version="3.13").pip_install(["fastapi", "pydantic", "uvicorn", "modal", "dotenv","httpx"])
image = image.add_local_file(local_path=os.path.join(os.path.dirname(__file__), "repl.py"), remote_path="/root/repl.py")
image = image.add_local_file(local_path=os.path.join(os.path.dirname(__file__), "repl_server.py"), remote_path="/root/repl_server.py")


app = App("repl-http-mcp", image=image)

@app.function(image=image, secrets=[Secret.from_dotenv()]) # set .env files in modal func, so can call sandboxes from a function
@modal.asgi_app()
def fastapi_app():
    from repl import Repl, ReplMCPExecResponse
    #state dicts for auto-stop functionality
    aliveRepls: Dict[str, Repl] =  Dict.from_name("aliveRepls", create_if_missing=True)
    replTimeouts: Dict[str, int] = Dict.from_name("replTimeouts", create_if_missing=True)
    replKillTimers: Dict[str, asyncio.TimerHandle] =  {} # TimerHandle not cloudpickle-able, but does not need to be stored in cloud
    replSnapshots: Dict[str, str] = Dict.from_name("replSnapshots", create_if_missing=True)

    web_app = fastapi.FastAPI()

    #create repl endpoint
    @web_app.post("/create_repl", status_code=status.HTTP_201_CREATED)
    async def create_repl(request: ReplMCPCreateRequest) -> ReplMCPCreateResponse:
        try:
            request.packages.extend(["fastapi", "pydantic", "uvicorn"])
            repl = await Repl.create(request.python_version, request.port, request.packages)
            aliveRepls[repl.id] = repl
            replTimeouts[repl.id] = request.timeout
            reset_repl_timer(repl.id)
            logger.info(f"Repl {repl.id} created with timeout of {replTimeouts[repl.id]} seconds")
            return ReplMCPCreateResponse(repl_id=repl.id)
        except Exception as e:
            logger.error(f"Error creating repl: {repr(e)}")
            raise HTTPException(status_code=500, detail=repr(e))

    @web_app.post("/exec", status_code=status.HTTP_200_OK)
    async def exec_cmd(request: ReplMCPExecRequest) -> ReplMCPExecResponse:
        try:
            repl = await get_repl(request.repl_id)
            commands = Repl.parse_command(request.command)
        except ValueError:
            logger.error(f"Repl {request.repl_id} not found")
            raise HTTPException(status_code=400, detail=f"Repl {request.repl_id} not found")
        try:
            response = await repl.run(commands)
            reset_repl_timer(repl.id)
            return response
        except HTTPException as e:
            logger.error(f"Error executing command {request.command} for repl {repl.id}: {e}")
            raise HTTPException(status_code=e.status_code, detail=e.detail)

    async def get_repl(repl_id: str) -> Repl:
        if repl_id in aliveRepls:
            logger.info(f"Repl {repl_id} found in aliveRepls")
            return aliveRepls[repl_id]
        elif repl_id in replSnapshots:
            logger.info(f"Recreating repl {repl_id} from snapshot")
            repl = await Repl.from_snapshot(replSnapshots[repl_id], repl_id)
            aliveRepls[repl.id] = repl
            del replSnapshots[repl_id]
            return repl
        logger.error(f"Repl {repl_id} not found")
        raise ValueError(f"Repl {repl_id} not found")

    def terminate_repl(repl_id: str) -> None:
        try:
            logger.info(f"Terminating repl {repl_id}")
            if repl_id in replKillTimers:
                replKillTimers[repl_id].cancel()
                replKillTimers.pop(repl_id, None)
            if repl_id not in aliveRepls:
                return
            repl = aliveRepls[repl_id]
            snapshot_id = repl.kill()
            replSnapshots[repl_id] = snapshot_id
            del aliveRepls[repl_id]
            logger.info(f"Repl {repl_id} terminated")
        except KeyError as e:
            logger.error(f"KeyError {repr(e)} for repl {repl_id}")
        except Exception as e:
            logger.error(f"Exception {repr(e)} for repl {repl_id}")

    def reset_repl_timer(repl_id: str) -> None:
        if repl_id in replTimeouts:
            if repl_id in replKillTimers:
                replKillTimers[repl_id].cancel()
            loop = asyncio.get_running_loop()
            replKillTimers[repl_id] = loop.call_later(replTimeouts[repl_id], terminate_repl, repl_id)
    return web_app

