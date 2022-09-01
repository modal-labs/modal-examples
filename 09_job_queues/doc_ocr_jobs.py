# ---
# integration-test: false
# ---
# # Document OCR Job Queue
#
# This tutorial shows you how to use Modal as an infinitely scalable job queue
# that can service async tasks from a web app. For the purpose of this tutorial,
# we've also built a [Modal serverless web app that submits tasks to the handler defined 
# here](/docs/examples/09_job_queues/doc_ocr_frontend), but note that you don't necessarily
# need to have your web app running on Modal as well - it can be any Python application, 
# such as a regular Django app running on Kubernetes.
# 
# Our job queue will handle a single task: running OCR transcription for a given image of a receipt.
# We'll make use of a pre-trained Document Understanding model using the 
# [donut](https://github.com/clovaai/donut) package to accomplish this.

# ## Define a Stub
#
# Let's first import `modal` and define a [`Stub`](/docs/reference/modal.Stub). Later, we'll use the name provided
# for our `Stub` to find it from our web app, and submit tasks to it.

import modal

stub = modal.Stub("doc_ocr_jobs")

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

    # Use donut fine-tuned on an OCR dataset.
    task_prompt = f"<s_cord-v2>"
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
# app, take a look at the [receipt parser frontend](/docs/examples/09_job_queues/doc_ocr_frontend)
# tutorial.

# ## Run manually
# 
# We can also trigger `parse_receipt` manually for easier debugging. To try it out, you can find some 
# example receipts [here](https://drive.google.com/drive/folders/1S2D1gXd4YIft4a5wDtW99jfl38e85ouW).

if __name__ == "__main__":
    with stub.run():
        with open("./receipt.png", "rb") as f:
            image = f.read()
            print(parse_receipt(image))
