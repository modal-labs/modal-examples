# ---
# deploy: true
# cmd: ["modal", "serve", "07_web_endpoints/fasthtml_app.py"]
# ---

# # Deploy a FastHTML app with Modal

# This example shows how you can deploy a FastHTML app with Modal.
# [FastHTML](https://www.fastht.ml/) is a Python library built on top of [HTMX](https://htmx.org/)
# which allows you to create entire web applications using only Python.
#
# Our example is a multiplayer checkbox game, inspired by [1 Million Checkboxes](https://onemillioncheckboxes.com/).
# We think it makes for a great demonstration of how you can build interactive, stateful web apps with FastHTML on Modal.

import time
from pathlib import Path, PurePosixPath
from threading import Lock
from uuid import uuid4
import os

import modal

app = modal.App("example-fasthtml")
db = modal.Dict.from_name("example-fasthtml-db", create_if_missing=True)

css_path_local = Path(__file__).parent / "fasthtml_app.css"
css_path_remote = PurePosixPath("/assets/fasthtml_app.css")

N_CHECKBOXES = 10_000  # feel free to increase, if you dare!

@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install(
        "python-fasthtml==0.6.9", "inflect~=7.4.0"
    ),
    concurrency_limit=1,  # we currently maintain state in memory, so we restrict the server to one worker
    mounts=[modal.Mount.from_local_file(css_path_local, remote_path=css_path_remote)],
)
@modal.asgi_app()
def web():
    import fasthtml.common as fh
    import inflect

    # Connected clients are tracked in-memory
    clients = {}
    clients_mutex = Lock()

    # We keep all checkbox states in memory during operation, and persist to modal dict across restarts
    checkboxes = db.get("checkboxes", [])
    checkbox_mutex = Lock()
    
    if len(checkboxes) == N_CHECKBOXES:
        print("Restored checkbox state from previous session.")
    else:
        print("Initializing checkbox state.")
        checkboxes = [False] * N_CHECKBOXES
    
    def on_shutdown():
        # Handle the shutdown event by persisting current state to modal dict
        with checkbox_mutex:
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
    def get():
        # register a new client
        client = Client()
        with clients_mutex:
            clients[client.id] = client

        # get current state of all checkboxes
        with checkbox_mutex:
            checkbox_array = [
                fh.CheckboxX(
                    id=f"cb-{i}",
                    checked=val,
                    # when clicked, that checkbox will send a POST request to the server with its index
                    hx_post=f"/checkbox/toggle/{i}/{client.id}", 
                )
                for i, val in enumerate(checkboxes)
            ]

        return (
            fh.Title(f"{N_CHECKBOXES // 1000}k Checkboxes"),
            fh.Main(
                fh.H1(
                    f"{inflect.engine().number_to_words(N_CHECKBOXES).title()} Checkboxes"
                ),
                fh.Div(
                    *checkbox_array,
                    id="checkbox-array",
                ),
                cls="container",
                # use HTMX to poll for diffs to apply
                hx_trigger="every 20s",  # poll every second
                hx_get=f"/diffs/{client.id}",  # call the diffs endpoint
                hx_swap="none",  # don't replace the entire page
            ),
        )

    # users submitting checkbox toggles
    @app.post("/checkbox/toggle/{i}/{client_id}")
    def toggle(i: int, client_id: str):
        with checkbox_mutex:
            checkboxes[i] = not checkboxes[i]

        with clients_mutex:
            expired = []
            for client in clients.values():
                if client.id == client_id:
                    # ignore self; we keep our own diffs
                    continue

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
    def diffs(client_id: str):
        # we use the `hx_swap_oob='true'` feature to
        # push updates only for the checkboxes that changed
        with clients_mutex:
            client = clients.get(client_id, None)
            if client is None or len(client.diffs) == 0:
                return

            client.heartbeat()
            diffs = client.pull_diffs()

        with checkbox_mutex:
            diff_array = [
                fh.CheckboxX(
                    id=f"cb-{i}",
                    checked=checkboxes[i],
                    hx_post=f"/checkbox/toggle/{i}/{client_id}",
                    hx_swap_oob="true",  # this allows us to push updates to arbitrary checkboxes matching the id
                )
                for i in diffs
            ]

        return diff_array

    return app


 #Class for tracking state to push out to connected clients
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
        if i in self.diffs:
            # two toggles are equivalent to zero, so we just cancel the diff
            self.diffs.remove(i)
        else:
            self.diffs.append(i)

    def pull_diffs(self):
        # return a copy of the diffs and clear them
        diffs = self.diffs
        self.diffs = []
        return diffs