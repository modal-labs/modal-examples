# ---
# integration-test: false
# ---
import modal

stub = modal.Stub("receipt_parser_jobs")

volume = modal.SharedVolume().persist("receipt_parser_model_vol")
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
    print("Result: ", output)

    return output


if __name__ == "__main__":
    with stub.run():
        with open("./receipt.png", "rb") as f:
            image = f.read()
            print(parse_receipt(image))
