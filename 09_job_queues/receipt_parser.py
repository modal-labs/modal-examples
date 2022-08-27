import fastapi
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import modal
from modal.functions import FunctionCall
from pathlib import Path

stub = modal.Stub()

web_app = fastapi.FastAPI()

assets_path = Path(__file__).parent / "receipt_parser_spa"

volume = modal.SharedVolume().persist("stable-diff-model-vol")
CACHE_PATH = "/root/model_cache"

@stub.function(
    gpu=True,
    image=modal.DebianSlim().pip_install(["donut-python"]),
    shared_volumes={CACHE_PATH: volume},
    retries=3,
)
def parse_receipt(image: bytes):
    from PIL import Image
    from donut import DonutModel
    import torch
    import io

    task_prompt = f"<s_cord-v2>"

    pretrained_model = DonutModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2", cache_dir=CACHE_PATH)

    pretrained_model.half()
    device = torch.device("cuda")
    pretrained_model.to(device)

    input_img = Image.open(io.BytesIO(image))
    output = pretrained_model.inference(image=input_img, prompt=task_prompt)["predictions"][0]
    print(output)

    return output


# @web_app.get("/factors")
# async def web_submit(request: fastapi.Request, number: int):
#     call = factor_number.submit(number)  # returns a FunctionCall without waiting for result
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
    def parse(body: dict = fastapi.Body(...)):
        # message = body["message"]
        # chat_id = body.get("id")
        # id, response = generate_response(message, chat_id)
        # return JSONResponse({"id": id, "response": response})
        return {}

    app.mount("/", StaticFiles(directory="/assets", html=True))
    return app


if __name__ == "__main__":
    stub.run_forever()
    # with stub.run():
    #     with open("./receipt.png", "rb") as f:
    #         image = f.read()
    #         parse_receipt(image)