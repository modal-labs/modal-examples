# ---
# pytest: false
# ---
"""Application serving logic for the CodeLangChain agent."""

import agent
import modal
from agent import app, create_sandbox
from fastapi import FastAPI, responses
from fastapi.middleware.cors import CORSMiddleware

# create a FastAPI app
web_app = FastAPI(
    title="CodeLangChain Server",
    version="1.0",
    description="Answers questions about Python programming.",
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


# host it on Modal
@app.function(keep_warm=1)
@modal.asgi_app()
def serve():
    from langchain_core.runnables import RunnableLambda
    from langserve import add_routes

    def inp(question: str) -> dict:
        return {"keys": {"question": question, "iterations": 0}}

    def out(state: dict) -> str:
        if "finish" in state:
            return state["finish"]["keys"]["response"]
        elif len(state) > 0 and "finish" in state[-1]:
            return state[-1]["finish"]["keys"]["response"]
        else:
            return str(state)

    graph = agent.construct_graph(create_sandbox(), debug=False).compile()

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

    return web_app
