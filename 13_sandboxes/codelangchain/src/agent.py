"""This module defines our agent and attaches it to the Modal App.

Our agent is defined as a graph: a collection of nodes and edges,
where nodes represent actions and edges represent transitions between actions.

The meat of the logic is therefore in the edges and nodes modules.

We have a very simple "context-stuffing" retrieval approach in the retrieval module.
Replace this with something that retrieves your documentation and adjust the prompts accordingly.

You can test the agent from the command line with `modal run agent.py --question` followed by your query"""

from . import edges, nodes, retrieval
from .common import PYTHON_VERSION, app

default_question = (
    f"What are some new Python features in Python {PYTHON_VERSION}?"
)


@app.local_entrypoint()
def main(
    question: str = default_question,
    debug: bool = False,
):
    """Sends a question to the Pytohn code generation agent.

    Switch to debug mode for shorter context and smaller model."""
    if debug:
        if question == default_question:
            question = "gm king, how are you?"
    print(go.remote(question, debug=debug))


@app.function()
def go(
    question: str = default_question,
    debug: bool = False,
):
    """Compiles the Python code generation agent graph and runs it, returning the result."""
    graph = construct_graph(debug=debug)
    runnable = graph.compile()
    result = runnable.invoke(
        {"keys": {"question": question, "iterations": 0}},
        config={"recursion_limit": 50},
    )

    return result["keys"]["response"]


def construct_graph(debug=False):
    from langgraph.graph import StateGraph

    from .common import GraphState

    context = retrieval.retrieve_docs(debug=debug)

    graph = StateGraph(GraphState)

    # attach our nodes to the graph
    graph_nodes = nodes.Nodes(context, debug=debug)
    for key, value in graph_nodes.node_map.items():
        graph.add_node(key, value)

    # construct the graph by adding edges
    graph = edges.enrich(graph)

    # set the starting and ending nodes of the graph
    graph.set_entry_point(key="generate")
    graph.set_finish_point(key="finish")

    return graph
