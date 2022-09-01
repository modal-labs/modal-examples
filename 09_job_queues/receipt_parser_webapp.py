# # Receipt Parser Webapp
#
# This tutorial shows you how to use Modal to deploy a fully serverless
# React application with a FastAPI back-end. We're going to build a simple "Receipt Parser" webapp 
# that submits OCR transcription tasks asynchronously to the app defined in the [Job Queue tutorial]
# (/docs/guide/ex/receipt_parser_jobs), polls until the task is completed, and displays the results.

# that can service async tasks from a web app. For the purpose of this tutorial,
# we've also built a [Modal serverless web app that submits tasks to the handler defined 
# here](/docs/examples/09_job_queues/receipt_parser_frontend), but note that you don't necessarily
# need to have your web app running on Modal as well - it can be any Python application, 
# such as a regular Django app running on Kubernetes.
# 
# Our job queue will handle a single task: running OCR transcription for a given receipt image.
# We'll make use of a pre-trained Document Understanding model using the 
# [donut](https://github.com/clovaai/donut) package to accomplish this.

# ## Define a Stub
#
# Let's first import `modal` and define a [`Stub`](/docs/reference/modal.Stub). Later, we'll use the name provided
# for our `Stub` to find it from our web app, and submit tasks to it.
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
