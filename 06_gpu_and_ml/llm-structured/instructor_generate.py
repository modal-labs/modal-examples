# ---
# output-directory: "/tmp/instructor_generate"
# ---
# # Structured Data Extraction using `instructor`
#
# This example demonstrates how to use the `instructor` library to extract structured data from unstructured text.
#
# Structured output is a powerful, but under-appreciated feature of LLMs,
# because it both makes it easier to connect LLMs to other software
# and allows for the ingestion of unstructured data into structured databases.
#
# The unstructured data in this example is the code from the examples in the Modal examples repository --
# including this one. We'll extract the libraries and Modal features used in each example,
#
# We use this exact code to monitor the coverage of the examples
# and to make decisions about which examples to write next!
#
# The output includes a JSONL file containing, on each line, the metadata extracted from the code in one example.
# This  can be consumed downstream by other software systems, like a database,
# and some
#
#
# ## Environment setup
#
# We setup the environment our code will run in first.
# In Modal, we define environments via [container images](https://modal.com/docs/guide/custom-container),
# much like Docker images, by iteratively chaining together commands.
#
# This example also uses models from OpenAI, so if you want to run it yourself,
# you'll need to set up a Modal [`Secret`](https://modal.com/docs/guide/secrets)
# called `openai-secret` for your OpenAI API key.
from pathlib import Path
from typing import Optional

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "instructor==0.6.4", "openai~=1.13.3", "matplotlib~=3.8.3"
)
stub = modal.Stub(
    image=image, secrets=[modal.Secret.from_name("openai-secret")]
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
# saving that metadata and some visualizations.
#
# One fun thing to note is that we don't build the visualizations locally,
# so we don't need to have `matplotlib` installed on our local machine.
# Instead, we generate the figures remotely and then send back as images.
#
# We include the option to run `with_gpt4`, which gives much better results,
# but it is off by default because GPT-4 is also ~20x more expensive.


@stub.local_entrypoint()
def main(limit: int = 15, with_gpt4: bool = False):
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
                f"### {example.stem}\n" + Path(example.filename).read_text()
                for example in examples
            ],
            kwargs={"with_gpt4": with_gpt4},
        )

    # save the results to a local file
    results_path = Path("/tmp") / "instructor_generate" / "results.jsonl"
    results_dir = results_path.parent
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    if limit > 0:  # use --limit=0 to reuse past results
        print(f"writing results to {results_path}")
        with open(results_path, "w") as f:
            for result in results:
                print(result)
                f.write(result + "\n")

    # create a visualization of which Modal features examples use
    figure = create_features_figure.remote(results_path.read_text())
    figure_path = results_dir / "features_figure.png"
    with open(figure_path, "wb") as file:
        print(f"Saving figure to {figure_path}")
        file.write(figure)

    # create a visualization of which 3rd-party libraries examples use
    figure = create_libraries_figure.remote(results_path.read_text())
    figure_path = results_dir / "libraries_figure.png"
    with open(figure_path, "wb") as file:
        print(f"Saving figure to {figure_path}")
        file.write(figure)


# ## Extracting JSON from unstructured text with `instructor`
#
# The real meat of this example is here, in the `extract_example_metadata` function.
#
# TODO: write this up


@stub.function(concurrency_limit=3)  # watch those openai rate limits!
def extract_example_metadata(
    example_contents: Optional[str] = None, with_gpt4=False
):
    import instructor
    from openai import OpenAI
    from pydantic import BaseModel, Field

    if example_contents is None:
        example_contents = Path(__file__).read_text()

    client = instructor.patch(OpenAI())

    all_modal_features = modal.__all__

    class ExampleMetadata(BaseModel):
        """Metadata about an example from the Modal examples repo."""

        filename: str = Field(..., description="The filename of the example.")
        libraries: list[str] = Field(
            ...,
            description="The third-party libraries imported in the example, if any.",
        )
        modal_features: list[str] = Field(
            ...,
            description=f"The Modal features that are directly used in the example. The list of all possible Modal features is: {','.join(all_modal_features)}.",
        )

    model = "gpt-4-0125-preview" if with_gpt4 else "gpt-3.5-turbo-0125"

    example = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=1024,
        response_model=ExampleMetadata,
        messages=[
            {
                "role": "user",
                "content": f"Extract the metadata for this example.\n\n-----EXAMPLE BEGINS-----{example_contents}-----EXAMPLE ENDS-----\n\n",
            },
        ],
    )

    return example.model_dump_json()


# ## Addenda
#
# The rest of the code used in this example is not particularly interesting:
# some boilerplate matplotlib code to generate the figures,
# and a utility function to find all of the examples.


@stub.function()
def create_features_figure(results_text) -> bytes:
    """Create a figure that shows the Modal features used in each example."""
    import io
    import json
    from collections import defaultdict

    import matplotlib
    import matplotlib.font_manager
    import matplotlib.pyplot as plt

    plt.xkcd()
    matplotlib.rcParams["font.family"] = "sans-serif"
    metadata = [json.loads(line) for line in results_text.split("\n") if line]

    # construct a dictionary mapping features to their counts, sorted
    all_features = defaultdict(lambda: 0)
    for m in metadata:
        for feature in m["modal_features"]:
            all_features[feature] += 1

    all_features = dict(
        sorted(all_features.items(), key=lambda item: item[1], reverse=True)
    )

    fig, ax = plt.subplots(
        figsize=(4 + len(all_features) / 2, 4 + len(metadata) / 2)
    )
    ax.imshow(
        [
            [0 if feature in m["modal_features"] else 1 for m in metadata]
            for feature in all_features
        ],
        cmap="Greys",
    )
    ax.set_xticks(range(len(metadata)))
    ax.set_xticklabels(
        [Path(m["filename"]).stem for m in metadata], rotation=75, ha="right"
    )
    ax.vlines(
        range(len(metadata)),
        0 - 0.5,
        len(all_features),
        color="xkcd:bright green",
        linewidth=0.5,
    )
    ax.set_ylim(len(all_features) - 0.5, 0 - 0.5)
    ax.tick_params(axis="x", labelbottom=True, labeltop=False)
    ax.set_yticks(range(len(all_features)))
    ax.set_yticklabels(all_features)
    ax.set_title(
        "Which examples use which features of Modal?", fontsize="xx-large"
    )
    plt.tight_layout()

    # Saving plot as raw bytes to send back
    buf = io.BytesIO()

    fig.savefig(buf, format="png", dpi=288, bbox_inches="tight")

    buf.seek(0)

    return buf.getvalue()


@stub.function()
def create_libraries_figure(results_text) -> bytes:
    """Create a figure that shows the libraries used in each example."""
    import io
    import json
    import sys
    from collections import defaultdict

    import matplotlib
    import matplotlib.font_manager
    import matplotlib.pyplot as plt

    plt.xkcd()
    matplotlib.rcParams["font.family"] = "sans-serif"
    metadata = [json.loads(line) for line in results_text.split("\n") if line]

    # construct a dictionary mapping libraries to their counts, sorted
    all_libraries = defaultdict(lambda: 0)
    ignore_libs = set(sys.stdlib_module_names) | {"modal"}
    for m in metadata:
        for library in m["libraries"]:
            if library not in ignore_libs:
                all_libraries[library] += 1

    all_libraries = dict(
        sorted(all_libraries.items(), key=lambda item: item[1], reverse=True)
    )

    fig, ax = plt.subplots(
        figsize=(4 + len(all_libraries) / 2, 4 + len(metadata) / 2)
    )
    ax.imshow(
        [
            [0 if library in m["libraries"] else 1 for m in metadata]
            for library in all_libraries
        ],
        cmap="Greys",
    )
    ax.set_xticks(range(len(metadata)))
    ax.set_xticklabels(
        [Path(m["filename"]).stem for m in metadata], rotation=75, ha="right"
    )
    ax.vlines(
        range(len(metadata)),
        0 - 0.5,
        len(all_libraries),
        color="xkcd:bright green",
        linewidth=0.5,
    )
    ax.set_ylim(len(all_libraries) - 0.5, 0 - 0.5)
    ax.tick_params(axis="x", labelbottom=True, labeltop=False)
    ax.set_yticks(range(len(all_libraries)))
    ax.set_yticklabels(all_libraries)
    ax.set_title("Which examples use which libraries?", fontsize="xx-large")
    plt.tight_layout()

    # Saving plot as raw bytes to send back
    buf = io.BytesIO()

    fig.savefig(buf, format="png", dpi=288, bbox_inches="tight")

    buf.seek(0)

    return buf.getvalue()


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
        for example in example_utils.get_examples(silent=silent)
        if example.type != 2  # filter out non-code assets
    ]
    return examples
