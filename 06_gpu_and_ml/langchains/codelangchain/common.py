"""Shared information: image definitions and common utilities."""

import os
from typing import Dict, TypedDict

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "beautifulsoup4~=4.12.3",
    "langchain==0.1.11",
    "langgraph==0.0.26",
    "langchain_community==0.0.27",
    "langchain-openai==0.0.8",
    "langserve[all]==0.0.46",
)

agent_image = image.pip_install(
    "chromadb==0.4.24",
    "langchainhub==0.1.15",
    "faiss-cpu~=1.8.0",
    "tiktoken==0.6.0",
)

app = modal.App(
    "code-langchain",
    image=image,
    secrets=[
        modal.Secret.from_name("my-openai-secret"),
        modal.Secret.from_name("my-langsmith-secret"),
    ],
)


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]


os.environ["LANGCHAIN_PROJECT"] = "codelangchain"

COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}
