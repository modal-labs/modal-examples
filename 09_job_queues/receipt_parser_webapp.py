import fastapi
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import modal
from pathlib import Path

stub = modal.Stub()

web_app = fastapi.FastAPI()

assets_path = Path(__file__).parent / "receipt_parser_frontend"


# @web_app.get("/factors")
# async def web_submit(request: fastapi.Request, number: int): #     call = factor_number.submit(number)  # returns a FunctionCall without waiting for result
#     polling_url = request.url.replace(path="/result", query=f"function_id={call.object_id}")
#     return RedirectResponse(polling_url)


# @web_app.get("/result")
# async def web_poll(function_id: str):
#     function_call = FunctionCall.from_id(function_id)
#     try:
#         result = function_call.get(timeout=0)
#     except TimeoutError:
#         result = "not ready"

#     return result


@stub.asgi(mounts=[modal.Mount("/assets", local_dir=assets_path)])
def transformer():
    app = fastapi.FastAPI()

    @app.post("/parse")
    async def parse(request: Request):
        body = await request.form()
        print(body)
        # id, response = generate_response(message, chat_id)
        # return JSONResponse({"id": id, "response": response})
        return {}

    app.mount("/", StaticFiles(directory="/assets", html=True))
    return app


if __name__ == "__main__":
    stub.serve()