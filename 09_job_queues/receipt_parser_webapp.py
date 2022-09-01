import fastapi
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import modal
from pathlib import Path

stub = modal.Stub()

web_app = fastapi.FastAPI()

assets_path = Path(__file__).parent / "receipt_parser_frontend"

parse_receipt = modal.lookup("receipt_parser_jobs", "parse_receipt")


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
        form = await request.form()
        receipt = await form['receipt'].read()
        call = parse_receipt.submit(receipt)
        return { "function_id": call.object_id }

    app.mount("/", StaticFiles(directory="/assets", html=True))
    return app


if __name__ == "__main__":
    stub.serve()