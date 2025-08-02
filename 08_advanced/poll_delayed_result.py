# ---
# cmd: ["modal", "serve", "08_advanced/poll_delayed_result.py"]
# ---

# # Polling for a delayed result on Modal

# This example shows how you can poll for a delayed result on Modal.

# The function `factor_number` takes a number as input and returns the prime factors of the number. The function could take a long time to run, so we don't want to wait for the result in the web server.
# Instead, we return a URL that the client can poll to get the result.

import fastapi
import modal
from modal.functions import FunctionCall
from starlette.responses import HTMLResponse, RedirectResponse

app = modal.App("example-poll-delayed-result")

web_app = fastapi.FastAPI()


@app.function(image=modal.Image.debian_slim().pip_install("primefac"))
def factor_number(number):
    import primefac

    return list(primefac.primefac(number))  # could take a long time


@web_app.get("/")
async def index():
    return HTMLResponse(
        """
    <form method="get" action="/factors">
        Enter a number: <input name="number" />
        <input type="submit" value="Factorize!"/>
    </form>
    """
    )


@web_app.get("/factors")
async def web_submit(request: fastapi.Request, number: int):
    call = factor_number.spawn(
        number
    )  # returns a FunctionCall without waiting for result
    polling_url = request.url.replace(
        path="/result", query=f"function_id={call.object_id}"
    )
    return RedirectResponse(polling_url)


@web_app.get("/result")
async def web_poll(function_id: str):
    function_call = FunctionCall.from_id(function_id)
    try:
        result = function_call.get(timeout=0)
    except TimeoutError:
        result = "not ready"

    return result


@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app
