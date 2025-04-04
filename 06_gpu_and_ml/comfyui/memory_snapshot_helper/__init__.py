import os

from aiohttp import web
from server import PromptServer

# ------- API Endpoints -------


@PromptServer.instance.routes.post("/cuda/set_device")
async def set_current_device(request):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    return web.json_response({"status": "success"})


# Empty for ComfyUI node registration
NODE_CLASS_MAPPINGS = {}
