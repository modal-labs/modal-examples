# ---
# pytest: false
# cmd: ["modal", "serve", "13_sandboxes.codelangchain.langserve"]
# ---

# # Deploy LangChain and LangGraph applications with LangServe

# This code demonstrates how to deploy a
# [LangServe](https://python.langchain.com/docs/langserve/) application on Modal.
# LangServe makes it easy to wrap LangChain and LangGraph applications in a FastAPI server,
# and Modal makes it easy to deploy FastAPI servers.

# The LangGraph application that it serves is from our [sandboxed LLM coding agent example](https://modal.com/docs/examples/agent).

# You can find the code for the agent and several other code files associated with this example in the
# [`codelangchain` directory of our examples repo](https://github.com/modal-labs/modal-examples/tree/main/13_sandboxes/codelangchain).

import modal

from .agent import construct_graph, create_sandbox
from .src.common import image

app = modal.App("example-langserve")

image = image.pip_install("langserve[all]==0.3.0")


@app.function(
    image=image,
    secrets=[  # see the agent.py file for more information on Secrets
        modal.Secret.from_name(
            "openai-secret", required_keys=["OPENAI_API_KEY"]
        ),
        modal.Secret.from_name(
            "langsmith-secret", required_keys=["LANGCHAIN_API_KEY"]
        ),
    ],
)
@modal.asgi_app()
def serve():
    from fastapi import FastAPI, responses
    from fastapi.middleware.cors import CORSMiddleware
    from langchain_core.runnables import RunnableLambda
    from langserve import add_routes

    # create a FastAPI app
    web_app = FastAPI(
        title="CodeLangChain Server",
        version="1.0",
        description="Writes code and checks if it runs.",
    )

    # set all CORS enabled origins
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    def inp(question: str) -> dict:
        return {"keys": {"question": question, "iterations": 0}}

    def out(state: dict) -> str:
        if "finish" in state:
            return state["finish"]["keys"]["response"]
        elif len(state) > 0 and "finish" in state[-1]:
            return state[-1]["finish"]["keys"]["response"]
        else:
            return str(state)

    graph = construct_graph(create_sandbox(app), debug=False).compile()

    chain = RunnableLambda(inp) | graph | RunnableLambda(out)

    add_routes(
        web_app,
        chain,
        path="/codelangchain",
    )

    # redirect the root to the interactive playground
    @web_app.get("/")
    def redirect():
        return responses.RedirectResponse(url="/codelangchain/playground")

    # return the FastAPI app and Modal will deploy it for us
    return web_app
