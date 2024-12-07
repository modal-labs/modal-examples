# ---
# cmd: ["modal", "run", "13_sandboxes.codelangchain.agent", "--question", "Use gpt2 and transformers to generate text"]
# pytest: false
# env: {"MODAL_AUTOMOUNT": "True"}
# ---

# # Build a coding agent with Modal Sandboxes and LangGraph

# This example demonstrates how to build an LLM coding "agent" that can generate and execute Python code, using
# documentation from the web to inform its approach.

# Naturally, we use the agent to generate code that runs language models.

# The agent is built with [LangGraph](https://github.com/langchain-ai/langgraph), a library for building
# directed graphs of computation popular with AI agent developers,
# and uses models from the OpenAI API.

# ## Setup

import modal

from .src import edges, nodes, retrieval
from .src.common import COLOR, PYTHON_VERSION, image

# You will need two [Modal Secrets](https://modal.com/docs/guide/secrets) to run this example:
# one to access the OpenAI API and another to access the LangSmith API for logging the agent's behavior.

# To create them, head to the [Secrets dashboard](https://modal.com/secrets), select "Create new secret",
# and use the provided templates for OpenAI and LangSmith.

app = modal.App(
    "example-code-langchain",
    image=image,
    secrets=[
        modal.Secret.from_name(
            "openai-secret", required_keys=["OPENAI_API_KEY"]
        ),
        modal.Secret.from_name(
            "langsmith-secret", required_keys=["LANGCHAIN_API_KEY"]
        ),
    ],
)

# ## Creating a Sandbox

# We execute the agent's code in a Modal [Sandbox](https://modal.com/docs/guide/sandbox), which allows us to
# run arbitrary code in a safe environment. In this example, we will use the [`transformers`](https://huggingface.co/docs/transformers/index)
# library to generate text with a pre-trained model. Let's create a Sandbox with the necessary dependencies.


def create_sandbox(app) -> modal.Sandbox:
    # Change this image (and the retrieval logic in the retrieval module)
    # if you want the agent to give coding advice on other libraries!
    agent_image = modal.Image.debian_slim(
        python_version=PYTHON_VERSION
    ).pip_install(
        "torch==2.5.0",
        "transformers==4.46.0",
    )

    return modal.Sandbox.create(
        image=agent_image,
        timeout=60 * 10,  # 10 minutes
        app=app,
        # Modal sandboxes support GPUs!
        gpu="T4",
        # you can also pass secrets here -- note that the main app's secrets are not shared
    )


# We also need a way to run our code in the sandbox. For this, we'll write a simple wrapper
# around the Modal Sandox `exec` method. We use `exec` because it allows us to run code without spinning up a
# new container. And we can reuse the same container for multiple runs, preserving state.


def run(code: str, sb: modal.Sandbox) -> tuple[str, str]:
    print(
        f"{COLOR['HEADER']}📦: Running in sandbox{COLOR['ENDC']}",
        f"{COLOR['GREEN']}{code}{COLOR['ENDC']}",
        sep="\n",
    )

    exc = sb.exec("python", "-c", code)
    exc.wait()

    stdout = exc.stdout.read()
    stderr = exc.stderr.read()

    if exc.returncode != 0:
        print(
            f"{COLOR['HEADER']}📦: Failed with exitcode {sb.returncode}{COLOR['ENDC']}"
        )

    return stdout, stderr


# ## Constructing the agent's graph

# Now that we have the sandbox to execute code in, we can construct our agent's graph. Our graph is
# defined in the `edges` and `nodes` modules
# [associated with this example](https://github.com/modal-labs/modal-examples/tree/main/13_sandboxes/codelangchain).
# Nodes are actions that change the state. Edges are transitions between nodes.

# The idea is simple: we start at the node `generate`, which invokes the LLM to generate code based off documentation.
# The generated code is executed (in the sandbox) as part of an edge called `check_code_execution`
# and then the outputs are passed to the LLM for evaluation (the `evaluate_execution` node).
# If the LLM determines that the code has executed correctly -- which might mean that the code raised an exception! --
# we pass along the `decide_to_finish` edge and finish.


def construct_graph(sandbox: modal.Sandbox, debug: bool = False):
    from langgraph.graph import StateGraph

    from .src.common import GraphState

    # Crawl the transformers documentation to inform our code generation
    context = retrieval.retrieve_docs(debug=debug)

    graph = StateGraph(GraphState)

    # Attach our nodes to the graph
    graph_nodes = nodes.Nodes(context, sandbox, run, debug=debug)
    for key, value in graph_nodes.node_map.items():
        graph.add_node(key, value)

    # Construct the graph by adding edges
    graph = edges.enrich(graph)

    # Set the starting and ending nodes of the graph
    graph.set_entry_point(key="generate")
    graph.set_finish_point(key="finish")

    return graph


# We now set up the graph and compile it. See the `src` module for details
# on the content of the graph and the nodes we've defined.

DEFAULT_QUESTION = "How do I generate Python code using a pre-trained model from the transformers library?"


@app.function()
def go(
    question: str = DEFAULT_QUESTION,
    debug: bool = False,
):
    """Compiles the Python code generation agent graph and runs it, returning the result."""
    sb = create_sandbox(app)

    graph = construct_graph(sb, debug=debug)
    runnable = graph.compile()
    result = runnable.invoke(
        {"keys": {"question": question, "iterations": 0}},
        config={"recursion_limit": 50},
    )

    sb.terminate()

    return result["keys"]["response"]


# ## Running the Graph

# Now let's call the agent from the command line!

# We define a `local_entrypoint` that runs locally and triggers execution on Modal.

# You can invoke it by executing following command from a folder that contains the `codelangchain` directory
# [from our examples repo](https://github.com/modal-labs/modal-examples/tree/main/13_sandboxes/codelangchain):

# ```bash
# modal run codelangchain.agent --question "How do I run a pre-trained model from the transformers library?"
# ```


@app.local_entrypoint()
def main(
    question: str = DEFAULT_QUESTION,
    debug: bool = False,
):
    """Sends a question to the Python code generation agent.

    Switch to debug mode for shorter context and smaller model."""
    if debug:
        if question == DEFAULT_QUESTION:
            question = "hi there, how are you?"

    print(go.remote(question, debug=debug))


# If things are working properly, you should see output like the following:

# ```bash
# $ modal run agent.py --question "generate some cool output with transformers"
# ---DECISION: FINISH---
# ---FINISHING---
# To generate some cool output using transformers, we can use a pre-trained language model from the Hugging Face Transformers library. In this example, we'll use the GPT-2 model to generate text based on a given prompt. The GPT-2 model is a popular choice for text generation tasks due to its ability to produce coherent and contextually relevant text. We'll use the pipeline API from the Transformers library, which simplifies the process of using pre-trained models for various tasks, including text generation.
#
# from transformers import pipeline
# # Initialize the text generation pipeline with the GPT-2 model
# generator = pipeline('text-generation', model='gpt2')
#
# # Define a prompt for the model to generate text from
# prompt = "Once upon a time in a land far, far away"
#
# # Generate text using the model
# output = generator(prompt, max_length=50, num_return_sequences=1)
#
# # Print the generated text
# print(output[0]['generated_text'])
#
# Result of code execution:
# Once upon a time in a land far, far away, and still inhabited even after all the human race, there would be one God: a perfect universal God who has always been and will ever be worshipped. All His acts and deeds are immutable,
# ```
