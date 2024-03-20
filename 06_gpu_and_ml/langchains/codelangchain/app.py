import agent
import modal
from agent import nodes, stub
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

web_app = FastAPI(
    title="CodeLangChain Server",
    version="1.0",
    description="Answers questions about LangChain Expression Language (LCEL).",
)


# Set all CORS enabled origins
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@stub.function(keep_warm=1)
@modal.asgi_app()
def serve():
    from langchain_core.runnables import RunnableLambda
    from langserve import add_routes

    def inp(question: str) -> dict:
        return {"keys": {"question": question, "iterations": 0}}

    def out(state: dict) -> str:
        if "keys" in state:
            return state["keys"]["response"]
        elif "generate" in state:
            return nodes.extract_response(state["generate"])
        else:
            return str(state)

    graph = agent.construct_graph(debug=False).compile()

    chain = RunnableLambda(inp) | graph | RunnableLambda(out)

    add_routes(
        web_app,
        chain,
        path="/codelangchain",
    )

    return web_app
