# ---
# deploy: true
# mypy: ignore-errors
# ---

# # Create a web wrapper for job queue, submission, polling, & results

# This simple tutorial shows you how to create an API endpoint that you can use
# to poll the status of your request.

# Let's first import `modal` and define an [`App`](https://modal.com/docs/reference/modal.App).

import time

import modal

app = modal.App("web_job_queue_wrapper")

# Next, we'll create a dummy backend service, in reality you may plug an a LLM or Diffusion model here.
# We'll add artificial delays to simulate a cold boot and a long-running tasks.

@app.cls()
class BackendService():
    @modal.enter()
    def enter(self):
        print('begin cold booting')
        time.sleep(10)
        print('end cold booting')

    @modal.method()
    def run(self, input_val: str):
        print(f'begin run with {input_val}')
        time.sleep(5)
        return input_val[::-1] # reverse the string

# Then, we can define a web endpoint that will submit a request to the backend service
# as well as other API routes for polling or retrieving results.

# To submit jobs asynchronously, we can use ['spawn'](https://modal.com/docs/reference/modal.Function#spawn),
# which return a [`FunctionCall`](https://modal.com/docs/reference/modal.FunctionCall) object that represents
# the submitted job.
#
# Then we can poll results by checking the ['call graph'](https://modal.com/docs/reference/modal.call_graph)
# of the `FunctionCall` object.

@app.function(image=modal.Image.debian_slim().pip_install("fastapi[standard]"))
@modal.asgi_app()
@modal.concurrent(max_inputs=100)
def web_endpoint():
    from fastapi import FastAPI, Request
    web_app = FastAPI()

    service = BackendService()

    @web_app.post("/run")
    async def submit(request: Request):
        """Asynchronously submit a request to the backend service."""
        input_val = (await request.json())['input_val']
        fc = service.run.spawn(input_val)
        return {"request_id": fc.object_id}

    @web_app.get("/requests/{request_id}/status")
    async def status(request_id: str):
        """Get the status of the request from the call graph."""
        fc = modal.FunctionCall.from_id(request_id)
        fc_input_info = fc.get_call_graph()[0].children[0]
        assert fc_input_info.function_call_id == fc.object_id, "unexpected graph"
        return {"status": fc_input_info.status.name}

    @web_app.get("/requests/{request_id}")
    async def result(request_id: str):
        fc = modal.FunctionCall.from_id(request_id)
        return {'response': fc.get()}

    return web_app

# To test this you can do:
# ```bash
# modal serve web_job_queue_wrapper.py
# ```

# And then do `python hit.py` where hit.py is:
# ```bash
# import modal
# import requests
# import time
#
# workspace = modal.config._profile
# environment = modal.config.config.get("environment") or ""
# prefix = workspace + (f"-{environment}" if environment else "")
# url = f"https://{prefix}--web-job-queue-wrapper-web-endpoint-dev.modal.run"
# print(f"Built URL: {url}")
#
# print("submitting request")
# result = requests.post(f"{url}/run", json={"input_val": "Hello, world!"})
# assert result.status_code == 200
# request_id = result.json()['request_id']
#
# print(f"got request id: {request_id}, polling status")
# while True:
    # status = requests.get(f"{url}/requests/{request_id}/status")
    # if status.status_code == 200:
        # data = status.json()
        # if data['status'] == 'SUCCESS':
            # print("request completed successfully")
            # break
        # else:
            # print(f"request result is {data['status']}")
    # else:
        # print('poll failed:', status)
    # time.sleep(1)
#
# print("retrieving result")
# result = requests.get(f"{url}/requests/{request_id}")
# print(f"result is {result.json()}")
# print("done")
# ```
