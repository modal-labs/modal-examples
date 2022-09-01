# # Document OCR Webapp
#
# This tutorial shows you how to use Modal to deploy a fully serverless
# React + FastAPI application. We're going to build a simple "Receipt Parser" webapp 
# that submits OCR transcription tasks to a separate Modal app defined in the [Job Queue tutorial]
# (/docs/guide/ex/doc_ocr_jobs), polls until the task is completed, and displays the results.

# ## Basic setup
#
# Let's first import `modal` and define a [`Stub`](/docs/reference/modal.Stub). Later, we'll use the name provided
# for our `Stub` to find it from our web app, and submit tasks to it.
import fastapi
import fastapi.staticfiles

from modal.functions import FunctionCall

import modal
from pathlib import Path

stub = modal.Stub()

web_app = fastapi.FastAPI()

parse_receipt = modal.lookup("doc_ocr_jobs", "parse_receipt")

@web_app.post("/parse")
async def parse(request: fastapi.Request):
    form = await request.form()
    receipt = await form['receipt'].read()
    call = parse_receipt.submit(receipt)
    return { "call_id": call.object_id }

@web_app.get("/result/{call_id}")
async def poll_results(call_id: str):
    function_call = FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=0)
    except TimeoutError:
        return fastapi.responses.JSONResponse(status_code=202)

    return result

assets_path = Path(__file__).parent / "doc_ocr_frontend"

@stub.asgi(mounts=[modal.Mount("/assets", local_dir=assets_path)])
def transformer():
    web_app.mount("/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True))

    return web_app

if __name__ == "__main__":
    stub.serve()
