# ---
# cmd: ["modal", "serve", "10_integrations/mcp_server_stateless.py"]
# ---

# # Deploy a remote, stateless MCP server on Modal

# This example demonstrates how to deploy a simple [MCP
# server](https://modelcontextprotocol.io/) on Modal, that provides a tool to get the
# current date and time in a given timezone. It is a stateless MCP server, meaning that
# it does not store any state between requests, and uses the "streamable HTTP" transport
# type.

# It uses the [FastMCP library](https://github.com/jlowin/fastmcp) to create the MCP
# server, which is wrapped using FastAPI, and served with a [Modal web
# endpoint](https://modal.com/docs/guide/webhooks).


import modal

app = modal.App("example-mcp-server-stateless")

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "fastapi==0.115.14",
    "fastmcp==2.10.6",
    "pydantic==2.11.10",
)


# First, we create the MCP server itself using FastMCP, and add a tool to it that
# allows LLMs to get the current date and time in a given timezone.


def make_mcp_server():
    from fastmcp import FastMCP

    mcp = FastMCP("Date and Time MCP Server")

    @mcp.tool()
    async def current_date_and_time(timezone: str = "UTC") -> str:
        """Get the current date and time.

        Args:
            timezone: The timezone to get the date and time in (optional). Defaults to UTC.

        Returns:
            The current date and time in the given timezone, in ISO 8601 format.
        """
        from datetime import datetime
        from zoneinfo import ZoneInfo

        try:
            tz = ZoneInfo(timezone)
        except Exception:
            raise ValueError(
                f"Invalid timezone '{timezone}'. Please use a valid timezone like 'UTC', "
                "'America/New_York', or 'Europe/Stockholm'."
            )
        return datetime.now(tz).isoformat()

    return mcp


# We then use FastMCP to create a Starlette app with `streamable-http` as transport
# type, and set `stateless_http=True` to make it stateless.

# This will be mounted by the FastAPI app, which we deploy as a Modal web endpoint using
# [the `asgi_app` decorator](https://modal.com/docs/reference/modal.asgi_app):


@app.function(image=image)
@modal.asgi_app()
def web():
    """ASGI web endpoint for the MCP server"""
    from fastapi import FastAPI

    mcp = make_mcp_server()
    mcp_app = mcp.http_app(transport="streamable-http", stateless_http=True)

    fastapi_app = FastAPI(lifespan=mcp_app.router.lifespan_context)
    fastapi_app.mount("/", mcp_app, "mcp")

    return fastapi_app


# And we're done!

# ## Testing the MCP server

# Now you can [serve](https://modal.com/docs/reference/cli/serve#modal-serve) the MCP
# server by running:

# ```bash
# modal serve mcp_server_stateless.py
# ```

# Then open the [MCP inspector](https://github.com/modelcontextprotocol/inspector):

# ```bash
# npx @modelcontextprotocol/inspector
# ```

# Enter the URL of the MCP server that was printed by the `modal serve` command above,
# suffixed with `/mcp/` (so for example
# `https://modal-labs-examples--datetime-mcp-server-web-dev.modal.run/mcp/`). Also
# make sure to select "Streamable HTTP" as the "Transport Type".

# After connecting and clicking "List Tools" in the "Tools" tab you should see your
# `current_date_and_time` tool listed, and if you "Run Tool" it should give you the
# current date and time in UTC!

# To automatically test the MCP server, we spin up a client and have it list the tools.
# This test is executed by running the script with `modal run`:

# ```bash
# modal run mcp_server_stateless::test_tool
# ```


@app.function(image=image)
async def test_tool(tool_name: str | None = None):
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    if tool_name is None:
        tool_name = "current_date_and_time"

    transport = StreamableHttpTransport(url=f"{web.get_web_url()}/mcp/")
    client = Client(transport)

    async with client:
        tools = await client.list_tools()

    for tool in tools:
        print(tool)
        if tool.name == tool_name:
            return

    raise Exception(f"could not find tool {tool_name}")
