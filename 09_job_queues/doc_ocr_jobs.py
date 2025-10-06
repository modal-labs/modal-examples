# ---
# deploy: true
# mypy: ignore-errors
# ---

# # Run a job queue that turns documents into structured data with Datalab Marker

# This tutorial shows you how to use Modal as an infinitely scalable job queue
# that can service async tasks from a web app.

# Our job queue will handle a single task: converting images/PDFs into structured data.
# We'll use [Marker](https://github.com/datalab-to/marker) from [Datalab](https://www.datalab.to),
# which can convert images of documents or PDFs to Markdown, JSON, and HTML. Marker is an open-weights model;
# to learn more about commercial usage, see [here](https://github.com/datalab-to/marker?tab=readme-ov-file#commercial-usage).

# For the purpose of this tutorial, we've also built a [React + FastAPI web app on Modal](https://modal.com/docs/examples/doc_ocr_webapp)
# that works together with it, but note that you don't need a web app running on Modal
# to use this pattern. You can submit async tasks to Modal from any Python
# application (for example, a regular Django app running on Kubernetes).

# Try it out for yourself [here](https://modal-labs-examples--example-doc-ocr-webapp-wrapper.modal.run/).

# ## Define an App

# Let's first import `modal` and define an [`App`](https://modal.com/docs/reference/modal.App).
# Later, we'll use the name provided for our job queue App to find it from our web app and submit tasks to it.

from typing import Optional

import modal
from typing_extensions import Literal

app = modal.App("example-doc-ocr-jobs")

# We also define the dependencies we need by specifying an
# [Image](https://modal.com/docs/guide/images).

inference_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "marker-pdf[full]==1.9.3", "torch==2.8.0"
)

# ## Cache the pre-trained model on a Modal Volume

# We can obtain the pre-trained model we want to run from Datalab
# by using the Marker library.


def load_models():
    import marker.models

    print("loading models")

    return marker.models.create_model_dict()


# The `create_model_dict` function downloads model weights from Datalab's
# cloud storage (S3 bucket) if they aren't already present in the filesystem.
# However, in Modal's serverless environment, filesystems are ephemeral,
# so using this code alone would mean that models need to be downloaded
# many times (every time a new instance of our Function spins up).

# So instead, we create a Modal [Volume](https://modal.com/docs/guide/volumes)
# to store the models. Each Modal Volume is a durable filesystem that any Modal Function can access.
# You can read more about storing model weights on Modal in [our guide](https://modal.com/docs/guide/model-weights).

marker_cache_path = "/root/.cache/datalab/"
marker_cache_volume = modal.Volume.from_name(
    "marker-models-modal-demo", create_if_missing=True
)
marker_cache = {marker_cache_path: marker_cache_volume}

# ## Run Datalab Marker on Modal

# Now let's set up the actual inference.

# Using the [`@app.function`](https://modal.com/docs/reference/modal.App#function)
# decorator, we set up a Modal [Function](https://modal.com/docs/reference/modal.Function).
# We provide arguments to that decorator to customize the hardware, scaling, and other features
# of the Function.

# Here, we say that this Function should use NVIDIA L40S [GPUs](https://modal.com/docs/guide/gpu),
# automatically [retry](https://modal.com/docs/guide/retries#function-retries) failures up to 3 times,
# and have access to our [shared model cache](https://modal.com/docs/guide/volumes).

# Inside the Function, we write out our inference logic,
# which mostly involves configuring components provided by the `marker` library.


@app.function(gpu="l40s", retries=3, volumes=marker_cache, image=inference_image)
def parse_document(
    document: bytes,
    page_range: str | None = None,
    force_ocr: bool = False,
    paginate_output: bool = False,
    output_format: Literal["markdown", "html", "chunks", "json"] = "markdown",
    use_llm: bool = False,
) -> str | dict:
    """
    Args:
        document: Document data (PDF, JPG, PNG) as bytes.
        page_range: Specify which pages to process. Accepts comma-separated page numbers and ranges.
        force_ocr: Force OCR processing on the entire document, even for pages that might contain extractable text.
                    This will also format inline math properly.
        paginate_output: Paginates the output, using \n\n{PAGE_NUMBER} followed by - * 48, then \n\n
        output_format: Output format. Can be markdown, JSON, HTML, or chunks.
        use_llm: use an llm to improve the marker results.
    """
    from tempfile import NamedTemporaryFile

    import marker.config.parser
    import marker.converters.pdf
    import marker.output

    models = load_models()

    # Set up document "converter"
    config = {
        "page_range": page_range,
        "force_ocr": force_ocr,
        "paginate_output": paginate_output,
        "output_format": output_format,
        "use_llm": use_llm,
    }

    config_parser = marker.config.parser.ConfigParser(config)
    config_dict = config_parser.generate_config_dict()
    config_dict["pdftext_workers"] = 1

    converter = marker.converters.pdf.PdfConverter(
        config=config_dict,
        artifact_dict=models,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service() if use_llm else None,
    )

    # Run the converter on our document
    with NamedTemporaryFile(delete=False, mode="wb+") as temp_path:
        temp_path.write(document)
        rendered_output = converter(temp_path.name)

    # Format the output and return it
    if output_format == "json":
        result = rendered_output.model_dump_json()
    else:
        text, _, images = marker.output.text_from_rendered(rendered_output)

        result = text

    return result


# ## Testing and debugging remote code

# To make sure this code works, we want a way to kick the tires and debug it.

# We can run it on Modal, with no need to set up separate local testing,
# by adding a [`local_entrypoint`](https://modal.com/docs/reference/modal.App#local_entrypoint)
# that invokes the Function `.remote`ly.


@app.local_entrypoint()
def main(document_filename: Optional[str] = None):
    import urllib.request
    from pathlib import Path

    if document_filename is None:
        document_filename = Path(__file__).parent / "receipt.png"
    else:
        document_filename = Path(document_filename)

    if document_filename.exists():
        image = document_filename.read_bytes()
        print(f"running OCR on {document_filename}")
    else:
        document_url = "https://modal-cdn.com/cdnbot/Brandys-walmart-receipt-8g68_a_hk_f9c25fce.webp"
        print(f"running OCR on sample from URL {document_url}")
        request = urllib.request.Request(document_url)
        with urllib.request.urlopen(request) as response:
            image = response.read()
    print(parse_document.remote(image, output_format="html"))


# You can then run this from the command line with:

# ```shell
# modal run doc_ocr_jobs.py
# ```

# ## Deploying the document conversion service

# Now that we have a Function, we can publish it by deploying the App:

# ```shell
# modal deploy doc_ocr_jobs.py
# ```

# Once it's published, we can [look up](https://modal.com/docs/guide/trigger-deployed-functions) this Function
# from another Python process and submit tasks to it:

# ```python
# fn = modal.Function.from_name("example-doc-ocr-jobs", "parse_document")
# fn.spawn(my_document)
# ```

# Modal will auto-scale to handle all the tasks queued, and
# then scale back down to 0 when there's no work left. To see how you could use this from a Python web
# app, take a look at the [receipt parser frontend](https://modal.com/docs/examples/doc_ocr_webapp)
# tutorial.
