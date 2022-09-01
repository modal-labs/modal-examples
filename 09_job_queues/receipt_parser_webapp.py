import fastapi
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from modal.functions import FunctionCall

import modal
from pathlib import Path

stub = modal.Stub()

web_app = fastapi.FastAPI()

assets_path = Path(__file__).parent / "receipt_parser_frontend"

parse_receipt = modal.lookup("receipt_parser_jobs", "parse_receipt")

@stub.asgi(mounts=[modal.Mount("/assets", local_dir=assets_path)])
def transformer():
    app = fastapi.FastAPI()

    @app.post("/parse")
    async def parse(request: Request):
        form = await request.form()
        receipt = await form['receipt'].read()
        call = parse_receipt.submit(receipt)
        return { "call_id": call.object_id }

    @app.get("/result/{call_id}")
    async def poll_results(call_id: str):
        function_call = FunctionCall.from_id(call_id)
        try:
            result = function_call.get(timeout=0)
        except TimeoutError:
            return JSONResponse(status_code=202)

        return result

    app.mount("/", StaticFiles(directory="/assets", html=True))
    return app


if __name__ == "__main__":
    stub.serve()