# ---
# integration-test: false
# ---
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
# In order for the model to be loaded only once per container, we put the model in a class and use a custom `__enter__`.


class Model:
    def __enter__(self):
        from donut import DonutModel
        import torch

        # Use donut fine-tuned on an OCR dataset.
        self.pretrained_model = DonutModel.from_pretrained(
            "naver-clova-ix/donut-base-finetuned-cord-v2", cache_dir=CACHE_PATH
        )

        # Initialize model.
        self.pretrained_model.half()
        device = torch.device("cuda")
        self.pretrained_model.to(device)

    @stub.function(
        gpu=True,
        image=modal.DebianSlim().pip_install(["donut-python"]),
        shared_volumes={CACHE_PATH: volume},
        retries=3,
    )
    def parse_receipt(self, image: bytes):
        from PIL import Image
        import io

        # Run inference.
        input_img = Image.open(io.BytesIO(image))
        task_prompt = "<s_cord-v2>"
        output = self.pretrained_model.inference(image=input_img, prompt=task_prompt)[
            "predictions"
        ][0]
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
    with stub.run():
        with open("./receipt.png", "rb") as f:
            image = f.read()
            print(Model().parse_receipt(image))
