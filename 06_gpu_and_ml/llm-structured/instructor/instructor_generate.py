# ---
# output-directory: "/tmp/instructor_generate"
# ---
# # Structured Data Extraction using `instructor`
#
# This example demonstrates how to use the `instructor` library to extract structured data from unstructured text.
#
# Structured output is a powerful but under-appreciated feature of LLMs,
# because it makes it easier to connect LLMs to other software,
# for example enabling the ingestion of unstructured data into structured databases.
#
# The unstructured data in this example is the code from the examples in the Modal examples repository --
# including this one!
#
# We use this exact code to monitor the coverage of the examples
# and to make decisions about which examples to write next!
#
# The output includes a JSONL file containing, on each line, the metadata extracted from the code in one example.
# This can be consumed downstream by other software systems, like a database or a dashboard.
#
# We include in this folder a Jupyter notebook with some basic analyses.
#
# ## Environment setup
#
# We setup the environment our code will run in first.
# In Modal, we define environments via [container images](https://modal.com/docs/guide/custom-container),
# much like Docker images, by iteratively chaining together commands.
#
# This example also uses models from Anthropic, so if you want to run it yourself,
# you'll need to set up a Modal [`Secret`](https://modal.com/docs/guide/secrets)
# called `my-anthropic-secret` for your OpenAI API key.
from pathlib import Path
from typing import Literal, Optional

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "instructor~=1.0.0", "anthropic~=0.23.1", "matplotlib~=3.8.3"
)

stub = modal.Stub(
    image=image, secrets=[modal.Secret.from_name("my-anthropic-secret")]
)


# ## The overall flow
#
# We'll run the example by calling `modal run instructor_generate.py` from the command line.
#
# When we invoke `modal run` on a Python file, we run the function
# marked with `@stub.local_entrypoint`.
#
# This is the only code that runs locally -- it coordinates
# the activity of the rest of our code, which runs in Modal's cloud.
#
# The logic is fairly simple: collect up the code for our examples,
# and then use `instructor` to extract metadata from them,
# which we then write to a file.
#
# By default, the language model is Claude 3 Haiku, the smallest model
# in the Claude 3 family.  We include the option to run `with_opus`,
# which gives much better results, but it is off by default because
# Opus is also ~60x more expensive, at ~$30 per million tokens.


@stub.local_entrypoint()
def main(limit: int = 15, with_opus: bool = False):
    # find all of the examples in the repo
    examples = get_examples()
    # optionally limit the number of examples we process
    if limit == 1:
        examples = [None]  # just run on this example
    else:
        examples = examples[:limit]
    if examples:
        # use Modal to map our extraction function over the examples concurrently
        results = extract_example_metadata.map(
            [
                f"{example.stem}\n" + Path(example.filename).read_text()
                if example
                else None
                for example in examples
            ],
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


# ## Extracting JSON from unstructured text with `instructor`
#
# The real meat of this example is here, in the `extract_example_metadata` function.
#
# TODO: write this up
# TODO: refactor classes out of this function, explain separately


@stub.function(concurrency_limit=5)  # watch those rate limits!
def extract_example_metadata(
    example_contents: Optional[str] = None,
    filename: Optional[str] = None,
    with_opus=False,
):
    import instructor
    from anthropic import Anthropic
    from pydantic import BaseModel, Field

    if example_contents is None:
        example_contents = Path(__file__).read_text()
        filename = Path(__file__).name

    class ExampleMetadataExtraction(BaseModel):
        """Extracted metadata about an example from the Modal examples repo."""

        summary: str = Field(..., description="A brief summary of the example.")
        has_thorough_explanation: bool = Field(
            ...,
            description="The example contains, in the form of inline comments with markdown formatting, a thorough explanation of what the code does.",
        )
        domains: list[
            Literal[
                "artificial_intelligence",
                "machine_learning",
                "data_science",
                "web_serving",
                "parallel_computing",
            ]
        ] = Field(..., description="The")
        difficulty: Literal[1, 2, 3, 4, 5] = Field(
            ...,
            description="The difficulty of the example, from 1 to 5. An example that uses only one or two basic Modal features and is understandable by a professional Python developer familiar with the basics of the relevant domains is a 1, while an example that uses many Modal features and uses advanced Python features like async generator coroutines or metaclasses is a 5.",
        )
        freshness: float = Field(
            ...,
            description="The freshness of the example, from 0 to 1. This is relative to your knowledge cutoff. Examples are less fresh if they use older libraries and tools.",
        )

    class ExampleMetadata(ExampleMetadataExtraction):
        """Metadata about an example from the Modal examples repo."""

        filename: str = Field(..., description="The filename of the example.")

    client = instructor.from_anthropic(Anthropic())

    model = "claude-3-opus-20240229" if with_opus else "claude-3-haiku-20240307"

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

    full_metadata = ExampleMetadata(
        **extracted_metadata.dict(), filename=filename
    )

    return full_metadata.model_dump_json()


# ## Addenda
#
# The rest of the code used in this example is not particularly interesting:
# some boilerplate matplotlib code to generate the figures,
# and a utility function to find all of the examples.


def get_examples(silent=True):
    """Find all of the examples using a utility from this repo.

    We use importlib to avoid the need to define the repo as a package."""
    import importlib

    examples_root = Path(__file__).parent.parent.parent.parent
    spec = importlib.util.spec_from_file_location(
        "utils", f"{examples_root}/internal/utils.py"
    )
    example_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example_utils)
    examples = [
        example
        for example in example_utils.get_examples(silent=silent)
        if example.type != 2  # filter out non-code assets
    ]
    return examples
