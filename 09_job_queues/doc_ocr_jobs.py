# ---
# deploy: true
# ---
#
# # Document OCR job queue
#
# This tutorial shows you how to use Modal as an infinitely scalable job queue
# that can service async tasks from a web app. For the purpose of this tutorial,
# we've also built a [React + FastAPI web app on Modal](/docs/guide/ex/doc_ocr_webapp)
# that works together with it, but note that you don't need a web app running on Modal
# to use this pattern. You can submit async tasks to Modal from any Python
# application (for example, a regular Django app running on Kubernetes).
#
# Our job queue will handle a single task: running OCR transcription for images.
# We'll make use of a pre-trained Document Understanding model using the
# [donut](https://github.com/clovaai/donut) package to accomplish this. Try
# it out for yourself [here](https://aksh-at-doc-ocr-webapp-wrapper.modal.run).
#
# ![receipt parser frontend](./receipt_parser_frontend_2.jpg)

# ## Define a Stub
#
# Let's first import `modal` and define a [`Stub`](/docs/reference/modal.Stub). Later, we'll use the name provided
# for our `Stub` to find it from our web app, and submit tasks to it.

import urllib.request

import modal

stub = modal.Stub("example-doc-ocr-jobs")

# ## Model cache
#
# `donut` downloads the weights for pre-trained models to a local directory, if those weights don't already exist.
# To decrease start-up time, we want this download to happen just once, even across separate function invocations.
# To accomplish this, we use a [`SharedVolume`](/docs/guide/shared-volumes), a writable volume that can be attached
# to Modal functions and persisted across function runs.

volume = modal.SharedVolume().persist("doc_ocr_model_vol")
CACHE_PATH = "/root/model_cache"

# ## Handler function
#
# Now let's define our handler function. Using the [@stub.function](https://modal.com/docs/reference/modal.Stub#function)
# decorator, we set up a Modal [Function](/docs/reference/modal.Function) that uses GPUs,
# has a [`SharedVolume`](/docs/guide/shared-volumes) mount, runs on a [custom container image](/docs/guide/custom-container),
# and automatically [retries](/docs/guide/retries#function-retries) failures up to 3 times.


@stub.function(
    gpu="any",
    image=modal.Image.debian_slim().pip_install("donut-python==1.0.7", "transformers==4.21.3"),
    shared_volumes={CACHE_PATH: volume},
    retries=3,
)
def parse_receipt(image: bytes):
    import io

    import torch
    from donut import DonutModel
    from PIL import Image

    # Use donut fine-tuned on an OCR dataset.
    task_prompt = "<s_cord-v2>"
    pretrained_model = DonutModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2", cache_dir=CACHE_PATH)

    # Initialize model.
    pretrained_model.half()
    device = torch.device("cuda")
    pretrained_model.to(device)

    # Run inference.
    input_img = Image.open(io.BytesIO(image))
    output = pretrained_model.inference(image=input_img, prompt=task_prompt)["predictions"][0]
    print("Result: ", output)

    return output


# ## Deploy
#
# Now that we have a function, we can publish it by deploying the app:
#
# ```shell
# modal app deploy doc_ocr_jobs.py
# ```
#
# Once it's published, we can [look up](/docs/guide/sharing-functions) this function from another
# Python process and submit tasks to it:
#
# ```python
# fn = modal.lookup("doc_ocr_jobs", "parse_receipt")
# fn.submit(my_image)
# ```
#
# Modal will auto-scale to handle all the tasks queued, and
# then scale back down to 0 when there's no work left. To see how you could use this from a Python web
# app, take a look at the [receipt parser frontend](/docs/guide/ex/doc_ocr_webapp)
# tutorial.

# ## Run manually
#
# We can also trigger `parse_receipt` manually for easier debugging. To try it out, you can find some
# example receipts [here](https://drive.google.com/drive/folders/1S2D1gXd4YIft4a5wDtW99jfl38e85ouW).

if __name__ == "__main__":
    from pathlib import Path

    receipt_filename = Path(__file__).parent / "receipt.png"
    with stub.run():
        if receipt_filename.exists():
            with open(receipt_filename, "rb") as f:
                image = f.read()
        else:
            image = urllib.request.urlopen(
                "https://nwlc.org/wp-content/uploads/2022/01/Brandys-walmart-receipt-8.webp"
            ).read()
        print(parse_receipt.call(image))
