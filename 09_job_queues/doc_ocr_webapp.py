# ---
# deploy: true
# ---
#
# # Document OCR web app
#
# This tutorial shows you how to use Modal to deploy a fully serverless
# [React](https://reactjs.org/) + [FastAPI](https://fastapi.tiangolo.com/) application.
# We're going to build a simple "Receipt Parser" web app that submits OCR transcription
# tasks to a separate Modal app defined in the [Job Queue
# tutorial](/docs/guide/ex/doc_ocr_jobs), polls until the task is completed, and displays
# the results. Try it out for yourself
# [here](https://modal-labs-example-doc-ocr-webapp-wrapper.modal.run/).
#
# ![receipt parser frontend](./receipt_parser_frontend.jpg)

# ## Basic setup
#
# Let's get the imports out of the way and define a [`Stub`](/docs/reference/modal.Stub).

from pathlib import Path

import fastapi
import fastapi.staticfiles
import modal
import modal.aio

stub = modal.Stub("example-doc-ocr-webapp")

# Modal works with any [ASGI](/docs/guide/webhooks#serving-asgi-and-wsgi-apps) or
# [WSGI](/docs/guide/webhooks#wsgi) web framework. Here, we choose to use [FastAPI](https://fastapi.tiangolo.com/).

web_app = fastapi.FastAPI()

# ## Define endpoints
#
# We need two endpoints: one to accept an image and submit it to the Modal job queue,
# and another to poll for the results of the job.
#
# In `parse`, we're going to submit tasks to the function defined in the [Job
# Queue tutorial](/docs/guide/ex/doc_ocr_jobs), so we import it first using
# [`modal.aio_lookup`](/docs/guide/sharing-functions#calling-code-from-outside-modal).
#
# We call [`.submit()`](/docs/reference/modal.Function#submit) on the function handle
# we imported above, to kick off our function without blocking on the results. `submit` returns
# a unique ID for the function call, that we can use later to poll for its result.


@web_app.post("/parse")
async def parse(request: fastapi.Request):
    # Use aio_lookup since we're in an async context.
    parse_receipt = modal.lookup("example-doc-ocr-jobs", "parse_receipt")

    form = await request.form()
    receipt = await form["receipt"].read()  # type: ignore
    call = parse_receipt.submit(receipt)
    return {"call_id": call.object_id}


# `/result` uses the provided `call_id` to instantiate a `modal.FunctionCall` object, and attempt
# to get its result. If the call hasn't finished yet, we return a `202` status code, which indicates
# that the server is still working on the job.


@web_app.get("/result/{call_id}")
async def poll_results(call_id: str):
    from modal.functions import FunctionCall

    function_call = FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=0)
    except TimeoutError:
        return fastapi.responses.JSONResponse(content="", status_code=202)

    return result


# Finally, we mount the static files for our front-end. We've made [a simple React
# app](https://github.com/modal-labs/modal-examples/tree/main/09_job_queues/doc_ocr_frontend)
# that hits the two endpoints defined above. To package these files with our app, first
# we get the local assets path, and then create a modal [`Mount`](/docs/guide/local-data#mounting-directories)
# that mounts this directory at `/assets` inside our container. Then, we instruct FastAPI to [serve
# this static file directory](https://fastapi.tiangolo.com/tutorial/static-files/) at our root path.

assets_path = Path(__file__).parent / "doc_ocr_frontend"


@stub.asgi(mounts=[modal.Mount("/assets", local_dir=assets_path)])
def wrapper():
    web_app.mount("/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True))

    return web_app


# ## Deploy
#
# That's all! To deploy your application, run
#
# ```shell
# modal app deploy doc_ocr_webapp.py
# ```
#
# If successful, this will print a URL for your app, that you can navigate to from
# your browser 🎉 .
#
# ![receipt parser processed](./receipt_parser_frontend_2.jpg)
#
# ## Developing
#
# If desired, instead of deploying, we can [serve](/docs/guide/webhooks#developing-with-stubserve)
# our app ephemerally. In this case, Modal watches all the mounted files, and updates
# the app if anything changes.

if __name__ == "__main__":
    stub.serve()
