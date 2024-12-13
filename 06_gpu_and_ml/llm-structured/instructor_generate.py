# ---
# output-directory: "/tmp/instructor_generate"
# ---

# # Structured Data Extraction using `instructor`

# This example demonstrates how to use the `instructor` library to extract structured, schematized data from unstructured text.

# Structured output is a powerful but under-appreciated feature of LLMs.
# Structured output allows LLMs and multimodal models to connect to traditional software,
# for example enabling the ingestion of unstructured data like text files into structured databases.
# Applied properly, it makes them an extreme example of the [Robustness Principle](https://en.wikipedia.org/wiki/Robustness_principle)
# Jon Postel formulated for TCP: "Be conservative in what you send, be liberal in what you accept".

# The unstructured data used in this example code is the code from the examples in the Modal examples repository --
# including this example's code!

# The output includes a JSONL file containing, on each line, the metadata extracted from the code in one example.
# This can be consumed downstream by other software systems, like a database or a dashboard.
# We've used it to maintain and update our [examples repository](https://github.com/modal-labs/modal-examples).

# ## Environment setup

# We set up the environment our code will run in first.
# In Modal, we define environments via [container images](https://modal.com/docs/guide/custom-container),
# much like Docker images, by iteratively chaining together commands.

# Here there's just one command, installing `instructor` and the Python SDK for Anthropic's LLM API.

from pathlib import Path
from typing import Literal, Optional

import modal
from pydantic import BaseModel, Field

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "instructor~=1.0.0", "anthropic~=0.23.1"
)

# This example uses models from Anthropic, so if you want to run it yourself,
# you'll need an Anthropic API key and a Modal [`Secret`](https://modal.com/docs/guide/secrets)
# called `my-anthropic-secret` to hold share it with your Modal Functions.

app = modal.App(
    image=image,
    secrets=[
        modal.Secret.from_name(
            "anthropic-secret", required_keys=["ANTHROPIC_API_KEY"]
        )
    ],
)

# ## Running Modal functions from the command line

# We'll run the example by calling `modal run instructor_generate.py` from the command line.

# When we invoke `modal run` on a Python file, we run the function
# marked with `@app.local_entrypoint`.

# This is the only code that runs locally -- it coordinates
# the activity of the rest of our code, which runs in Modal's cloud.

# The logic is fairly simple: collect up the code for our examples,
# and then use `instructor` to extract metadata from them,
# which we then write to a file.

# By default, the language model is Claude 3 Haiku, the smallest model
# in the Claude 3 family.  We include the option to run `with_opus`,
# which gives much better results, but it is off by default because
# Opus is also ~60x more expensive, at ~$30 per million tokens.


@app.local_entrypoint()
def main(limit: int = 1, with_opus: bool = False):
    # find all of the examples in the repo
    examples = get_examples()
    # optionally limit the number of examples we process
    if limit == 1:
        examples = [None]  # just run on this example
    else:
        examples = examples[:limit]
    # use Modal to map our extraction function over the examples concurrently
    results = extract_example_metadata.map(
        (  # iterable of file contents
            Path(example.filename).read_text() if example else None
            for example in examples
        ),
        (  # iterable of filenames
            example.stem if example else None for example in examples
        ),
        kwargs={"with_opus": with_opus},
    )

    # save the results to a local file
    results_path = Path("/tmp") / "instructor_generate" / "results.jsonl"
    results_dir = results_path.parent
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    print(f"writing results to {results_path}")
    with open(results_path, "w") as f:
        for result in results:
            print(result)
            f.write(result + "\n")


# ## Extracting JSON from unstructured text with `instructor` and Pydantic

# The real meat of this example is in this section, in the `extract_example_metadata` function and its schemas.

# We define a schema for the data we want the LLM to extract, using Pydantic.
# Instructor ensures that the LLM's output matches this schema.

# We can use the type system provided by Python and Pydantic to express many useful features
# of the data we want to extract -- ranging from wide-open fields like a `str`ing-valued `summary`
# to constrained fields like `difficulty`, which can only take on value between 1 and 5.


class ExampleMetadataExtraction(BaseModel):
    """Extracted metadata about an example from the Modal examples repo."""

    summary: str = Field(..., description="A brief summary of the example.")
    has_thorough_explanation: bool = Field(
        ...,
        description="The example contains, in the form of inline comments with markdown formatting, a thorough explanation of what the code does.",
    )
    tags: list[
        Literal[
            "use-case-inference-lms",
            "use-case-inference-audio",
            "use-case-inference-images-video-3d",
            "use-case-finetuning",
            "use-case-job-queues-batch-processing",
            "use-case-sandboxed-code-execution",
        ]
    ] = Field(..., description="The use cases associated with the example")
    freshness: float = Field(
        ...,
        description="The freshness of the example, from 0 to 1. This is relative to your knowledge cutoff. Examples are less fresh if they use older libraries and tools.",
    )


# That schema describes the data to be extracted by the LLM, but not all data is best extracted by an LLM.
# For example, the filename is easily determined in software.

# So we inject that information into the output after the LLM has done its work. That necessitates
# an additional schema, which inherits from the first.


class ExampleMetadata(ExampleMetadataExtraction):
    """Metadata about an example from the Modal examples repo."""

    filename: Optional[str] = Field(
        ..., description="The filename of the example."
    )


# With these schemas in hand, it's straightforward to write the function that extracts the metadata.
# Note that we decorate it with `@app.function` to make it run on Modal.


@app.function(concurrency_limit=5)  # watch those LLM API rate limits!
def extract_example_metadata(
    example_contents: Optional[str] = None,
    filename: Optional[str] = None,
    with_opus=False,
):
    import instructor
    from anthropic import Anthropic

    # if no example is provided, use the contents of this example
    if example_contents is None:
        example_contents = Path(__file__).read_text()
        filename = Path(__file__).name

    client = instructor.from_anthropic(Anthropic())
    model = "claude-3-opus-20240229" if with_opus else "claude-3-haiku-20240307"

    # add the schema as the `response_model` argument in what otherwise looks like a normal LLM API call
    extracted_metadata = client.messages.create(
        model=model,
        temperature=0.0,
        max_tokens=1024,
        response_model=ExampleMetadataExtraction,
        messages=[
            {
                "role": "user",
                "content": f"Extract the metadata for this example.\n\n-----EXAMPLE BEGINS-----{example_contents}-----EXAMPLE ENDS-----\n\n",
            },
        ],
    )

    # inject the filename
    full_metadata = ExampleMetadata(
        **extracted_metadata.dict(), filename=filename
    )

    # return it as JSON
    return full_metadata.model_dump_json()


# ## Addenda

# The rest of the code used in this example is not particularly interesting:
# just a utility function to find all of the examples, which we invoke in the `local_entrypoint` above.


def get_examples(silent=True):
    """Find all of the examples using a utility from this repo.

    We use importlib to avoid the need to define the repo as a package."""
    import importlib

    examples_root = Path(__file__).parent.parent.parent
    spec = importlib.util.spec_from_file_location(
        "utils", f"{examples_root}/internal/utils.py"
    )
    example_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example_utils)
    examples = [
        example
        for example in example_utils.get_examples()
        if example.type != 2  # filter out non-code assets
    ]
    return examples
