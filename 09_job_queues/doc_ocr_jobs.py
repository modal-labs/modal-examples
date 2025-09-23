# ---
# deploy: true
# mypy: ignore-errors
# ---
# # Run a job queue for Datalab's Document Information Extraction

# This tutorial shows you how to use Modal as an infinitely scalable job queue
# that can service async tasks from a web app. For the purpose of this tutorial,
# we've also built a [React + FastAPI web app on Modal](https://modal.com/docs/examples/doc_ocr_webapp)
# that works together with it, but note that you don't need a web app running on Modal
# to use this pattern. You can submit async tasks to Modal from any Python
# application (for example, a regular Django app running on Kubernetes).

# Our job queue will handle a single task: running OCR transcription for images of receipts.
# We'll use [Marker](https://github.com/datalab-to/marker) from [Datalab](https://www.datalab.to)
# which can convert documents to Markdown, JSON and HTML.

# Try it out for yourself [here](https://modal-labs-examples--example-doc-ocr-webapp-wrapper.modal.run/).

# [![Webapp frontend](https://modal-cdn.com/doc_ocr_frontend.jpg)](https://modal-labs-examples--example-doc-ocr-webapp-wrapper.modal.run/)

# ## Define an App

# Let's first import `modal` and define an [`App`](https://modal.com/docs/reference/modal.App).
# Later, we'll use the name provided for our `App` to find it from our web app and submit tasks to it.

from typing import Optional

import modal

app_name = "example-doc-ocr-jobs"
app = modal.App(app_name)

# We also define the dependencies for our Function by specifying an
# [Image](https://modal.com/docs/guide/images).

inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["git", "wget"])
    .env({"TORCH_DEVICE": "cuda"})
    .pip_install(
        [
            "marker-pdf[full]==1.9.3",
            "torch==2.8.0",
        ]
    )
)

# ## Cache the pre-trained model on a Modal Volume

# We can obtain the pre-trained model we want to run from Datalab
# using the Marker library.

# The logic for loading the model based on this information
# is defined in the `setup` function below.

def setup():
    from marker.models import create_model_dict

    return create_model_dict()


# The `create_model_dict` function downloads the model weights from Datalab's
# cloud storage (S3 bucket) if it hasn't been downloaded before.
# But in Modal's serverless environment, filesystems are ephemeral,
# and so using this code alone would mean that models need to get downloaded
# on every request.

# So instead, we create a Modal [Volume](https://modal.com/docs/guide/volumes)
# to store the model -- a durable filesystem that any Modal Function can access.

MODEL_PATH_PREFIX = "/root/.cache/datalab/models"
markers_cache = modal.Volume.from_name(
    "marker-models-modal-demo", create_if_missing=True
)

# ## Run OCR inference on Modal by wrapping with `app.function`

# Now let's set up the actual OCR inference.

# Using the [`@app.function`](https://modal.com/docs/reference/modal.App#function)
# decorator, we set up a Modal [Function](https://modal.com/docs/reference/modal.Function).
# We provide arguments to that decorator to customize the hardware, scaling, and other features
# of the Function.

# Here, we say that this Function should use NVIDIA L40S [GPUs](https://modal.com/docs/guide/gpu),
# automatically [retry](https://modal.com/docs/guide/retries#function-retries) failures up to 3 times,
# and have access to our [shared model cache](https://modal.com/docs/guide/volumes).


@app.function(
    gpu="l40s",
    retries=3,
    volumes={MODEL_PATH_PREFIX: markers_cache},
    image=inference_image,
)
def parse_receipt(
    image: bytes,
    page_range: Optional[str] = None,
    force_ocr: Optional[bool] = False,
    paginate_output: bool = False,
    output_format: str = "markdown",
    use_llm: Optional[bool] = False,
) -> dict:
    from tempfile import NamedTemporaryFile
    import warnings
    from marker.converters.pdf import PdfConverter
    from marker.config.parser import ConfigParser
    from marker.settings import settings
    from marker.output import text_from_rendered
    from marker.models import create_model_dict

    models = setup()
    with NamedTemporaryFile(delete=False, mode="wb+") as temp_path:
        temp_path.write(image)
        # Configure conversion parameters
        config = {
            "filepath": temp_path,
            "page_range": page_range,
            "force_ocr": force_ocr,
            "paginate_output": paginate_output,
            "output_format": output_format,
            "use_llm": use_llm,
        }

        # Create converter
        config_parser = ConfigParser(config)
        config_dict = config_parser.generate_config_dict()
        config_dict["pdftext_workers"] = 1

        converter = PdfConverter(
            config=config_dict,
            artifact_dict=models,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service() if use_llm else None,
        )
        rendered_output = converter(temp_path.name)
        metadata = rendered_output.metadata

        # Extract content based on output format
        json_content = None
        html_content = None
        markdown_content = None

        if output_format == "json":
            # For JSON, return the structured data directly
            json_content = rendered_output.model_dump()
        else:
            text, _, images = text_from_rendered(rendered_output)

            # Assign to appropriate content field
            if output_format == "html":
                html_content = text
            else:
                markdown_content = text

        return {
            "success": True,
            "output_format": output_format,
            "json": json_content,
            "html": html_content,
            "markdown": markdown_content,
            "metadata": metadata,
            "page_count": len(metadata.get("page_stats", [])),
        }


# ## Deploy

# Now that we have a function, we can publish it by deploying the app:

# ```shell
# modal deploy doc_ocr_jobs.py
# ```

# Once it's published, we can [look up](https://modal.com/docs/guide/trigger-deployed-functions) this Function
# from another Python process and submit tasks to it:

# ```python
# fn = modal.Function.from_name("example-doc-ocr-jobs", "parse_receipt")
# fn.spawn(my_image)
# ```

# Modal will auto-scale to handle all the tasks queued, and
# then scale back down to 0 when there's no work left. To see how you could use this from a Python web
# app, take a look at the [receipt parser frontend](https://modal.com/docs/examples/doc_ocr_webapp)
# tutorial.

# ## Run manually

# We can also trigger `parse_receipt` manually for easier debugging:

# ```shell
# modal run doc_ocr_jobs
# ```

# To try it out, you can find some
# example receipts [here](https://drive.google.com/drive/folders/1S2D1gXd4YIft4a5wDtW99jfl38e85ouW).


@app.local_entrypoint()
def main(receipt_filename: Optional[str] = None):
    import urllib.request
    from pathlib import Path

    if receipt_filename is None:
        receipt_filename = Path(__file__).parent / "receipt.png"
    else:
        receipt_filename = Path(receipt_filename)

    if receipt_filename.exists():
        image = receipt_filename.read_bytes()
        print(f"running OCR on {receipt_filename}")
    else:
        receipt_url = "https://modal-cdn.com/cdnbot/Brandys-walmart-receipt-8g68_a_hk_f9c25fce.webp"
        request = urllib.request.Request(receipt_url)
        with urllib.request.urlopen(request) as response:
            image = response.read()
        print(f"running OCR on sample from URL {receipt_url}")
    print(parse_receipt.remote(image))
