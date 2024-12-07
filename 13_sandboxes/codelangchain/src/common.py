"""Shared information: image definitions and common utilities."""

import os
from typing import Any, Dict, TypedDict

import modal

PYTHON_VERSION = "3.11"

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(
        "beautifulsoup4~=4.12.3",
        "langchain==0.3.4",
        "langchain-core==0.3.12",
        "langgraph==0.2.39",
        "langchain-community==0.3.3",
        "langchain-openai==0.2.3",
    )
    .env({"LANGCHAIN_TRACING_V2": "true"})
)


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, Any]


os.environ["LANGCHAIN_PROJECT"] = "codelangchain"
os.environ["LANGCHAIN_TRACING"] = "true"

COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}
