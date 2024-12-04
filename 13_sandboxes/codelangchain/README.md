# Deploying code agents without all the agonizing pain

This example deploys a "code agent": a language model that can write and execute
code in a flexible control flow aimed at completing a task or goal.

It is implemented in LangChain, using the LangGraph library to structure the
agent and the LangServe framework to turn it into a FastAPI app.

We use Modal to turn that app into a web endpoint. We also use Modal to
"sandbox" the agent's code execution, so that it can't accidentally (or when
prompt injected!) damage the application by executing some inadvisable code.

Modal's Charles Frye and LangChain's Lance Martin did a
[walkthrough webinar](https://www.youtube.com/watch?v=X3yzWtAkaeo) explaining
the project's context and implementation. Check it out if you're curious!

## How to run

To run this app, you need to `pip install modal` and then create the following
[secrets](https://modal.com/docs/guide/secrets):

- `openai-secret` with an OpenAI API key, so that we can query OpenAI's models
  to power the agent,
- and `langsmith-secret` with a LangSmith API key, so that we can monitor the
  agent's behavior with LangSmith.

Head to the [secret creation dashboard](https://modal.com/secrets/) and follow
the instructions for each secret type.

Then, you can deploy the app with:

```bash
modal deploy codelangchain.py
```

Navigate to the URL that appears in the output and you'll be dropped into an
interactive "playground" interface where you can send queries to the agent and
receive responses. You should expect it to take about a minute to respond.

You can also navigate to the `/docs` path to see OpenAPI/Swagger docs, for
everything you'd need to see how to incorporate the agent into your downstream
applications via API requests.

When developing the app, use `modal serve codelangchain.py` to get a
hot-reloading server.

## Repo structure

The web application is defined in `codelangchain.py`.

It wraps the `agent.py` module, which contains the LangChain agent's definition.
To test the agent in isolation, run `modal run agent.py` in the terminal and
provide a `--question` about Python programming as input.

Because the agent is a graph, it is defined by specifying nodes and edges, which
are found in `nodes.py` and `edges.py`, respectively.

The retrieval logic is very simple: all of the data from the relevant docs is
retrieved and put at the beginning of the language model's prompt. You can find
it in `retrieval.py`.

The definition of the Modal container images and a few other shared utilities
can be found in `common.py`.
