# ---
# cmd: ["modal", "serve", "07_web_endpoints.fasthtml-checkboxes.fasthtml_checkboxes"]
# deploy: true
# mypy: ignore-errors
# ---

# # Deploy 100,000 multiplayer checkboxes on Modal with FastHTML

# [![Screenshot of FastHTML Checkboxes UI](./ui.png)](https://modal-labs-examples--example-checkboxes-web.modal.run)

# This example shows how you can deploy a multiplayer checkbox game with FastHTML on Modal.

# [FastHTML](https://www.fastht.ml/) is a Python library built on top of [HTMX](https://htmx.org/)
# which allows you to create entire web applications using only Python.
# For a simpler template for using FastHTML with Modal, check out
# [this example](https://modal.com/docs/examples/fasthtml_app).

# Our example is inspired by [1 Million Checkboxes](https://onemillioncheckboxes.com/).

import time
from asyncio import Lock
from pathlib import Path
from uuid import uuid4

import modal

from .constants import N_CHECKBOXES

app = modal.App("example-checkboxes")
db = modal.Dict.from_name("example-checkboxes-db", create_if_missing=True)

css_path_local = Path(__file__).parent / "styles.css"
css_path_remote = Path("/assets/styles.css")


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install(
        "python-fasthtml==0.6.9", "inflect~=7.4.0"
    ),
    concurrency_limit=1,  # we currently maintain state in memory, so we restrict the server to one worker
    mounts=[
        modal.Mount.from_local_file(css_path_local, remote_path=css_path_remote)
    ],
    allow_concurrent_inputs=1000,
)
@modal.asgi_app()
def web():
    import fasthtml.common as fh
    import inflect

    # Connected clients are tracked in-memory
    clients = {}
    clients_mutex = Lock()

    # We keep all checkbox fasthtml elements in memory during operation, and persist to modal dict across restarts
    checkboxes = db.get("checkboxes", [])
    checkbox_mutex = Lock()

    if len(checkboxes) == N_CHECKBOXES:
        print("Restored checkbox state from previous session.")
    else:
        print("Initializing checkbox state.")
        checkboxes = []
        for i in range(N_CHECKBOXES):
            checkboxes.append(
                fh.Input(
                    id=f"cb-{i}",
                    type="checkbox",
                    checked=False,
                    # when clicked, that checkbox will send a POST request to the server with its index
                    hx_post=f"/checkbox/toggle/{i}",
                    hx_swap_oob="true",  # allows us to later push diffs to arbitrary checkboxes by id
                )
            )

    async def on_shutdown():
        # Handle the shutdown event by persisting current state to modal dict
        async with checkbox_mutex:
            db["checkboxes"] = checkboxes
        print("Checkbox state persisted.")

    style = open(css_path_remote, "r").read()
    app, _ = fh.fast_app(
        # FastHTML uses the ASGI spec, which allows handling of shutdown events
        on_shutdown=[on_shutdown],
        hdrs=[fh.Style(style)],
    )

    # handler run on initial page load
    @app.get("/")
    async def get():
        # register a new client
        client = Client()
        async with clients_mutex:
            clients[client.id] = client

        return (
            fh.Title(f"{N_CHECKBOXES // 1000}k Checkboxes"),
            fh.Main(
                fh.H1(
                    f"{inflect.engine().number_to_words(N_CHECKBOXES).title()} Checkboxes"
                ),
                fh.Div(
                    *checkboxes,
                    id="checkbox-array",
                ),
                cls="container",
                # use HTMX to poll for diffs to apply
                hx_trigger="every 1s",  # poll every second
                hx_get=f"/diffs/{client.id}",  # call the diffs endpoint
                hx_swap="none",  # don't replace the entire page
            ),
        )

    # users submitting checkbox toggles
    @app.post("/checkbox/toggle/{i}")
    async def toggle(i: int):
        async with checkbox_mutex:
            cb = checkboxes[i]
            cb.checked = not cb.checked
            checkboxes[i] = cb

        async with clients_mutex:
            expired = []
            for client in clients.values():
                # clean up old clients
                if not client.is_active():
                    expired.append(client.id)

                # add diff to client for when they next poll
                client.add_diff(i)

            for client_id in expired:
                del clients[client_id]
        return

    # clients polling for any outstanding diffs
    @app.get("/diffs/{client_id}")
    async def diffs(client_id: str):
        # we use the `hx_swap_oob='true'` feature to
        # push updates only for the checkboxes that changed
        async with clients_mutex:
            client = clients.get(client_id, None)
            if client is None or len(client.diffs) == 0:
                return

            client.heartbeat()
            diffs = client.pull_diffs()

        async with checkbox_mutex:
            diff_array = [checkboxes[i] for i in diffs]

        return diff_array

    return app


# Class for tracking state to push out to connected clients
class Client:
    def __init__(self):
        self.id = str(uuid4())
        self.diffs = []
        self.inactive_deadline = time.time() + 30

    def is_active(self):
        return time.time() < self.inactive_deadline

    def heartbeat(self):
        self.inactive_deadline = time.time() + 30

    def add_diff(self, i):
        if i not in self.diffs:
            self.diffs.append(i)

    def pull_diffs(self):
        # return a copy of the diffs and clear them
        diffs = self.diffs
        self.diffs = []
        return diffs
