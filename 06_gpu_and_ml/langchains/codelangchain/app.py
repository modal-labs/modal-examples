import agent
import modal
from agent import stub
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

    def inp(question: str):
        return {"keys": {"question": question, "iterations": 0}}

    def out(state: dict) -> str:
        print(state)
        if "generate" in state:  # patch issue with langserve playground
            return "\n".join(
                str(elem) for elem in state["generate"]["keys"]["generation"]
            )
        return state["keys"]["response"]

    runnable = agent.construct_graph(debug=False).compile()

    chain = RunnableLambda(inp) | runnable | RunnableLambda(out)

    add_routes(
        web_app,
        chain,
        path="/codelangchain",
    )

    return web_app
