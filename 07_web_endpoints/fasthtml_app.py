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
# It's a great demonstration of of how you can build interactive, stateful web apps with FastHTML on Modal.

from threading import Lock
import time
from uuid import uuid4

import fasthtml.common as fh

import modal

app = modal.App("example-fasthtml")

N_CHECKBOXES = 10_000 # feel free to increase, if you dare!

@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install(
        "python-fasthtml==0.6.9"
    ),
    concurrency_limit=1, # we currently maintain state in memory, so we restrict the server to one worker
)
@modal.asgi_app() 
def web():
    
    app, _ = fh.fast_app()

    # our in-memory state for the checkboxes
    checkboxes = [False] * N_CHECKBOXES
    checkbox_mutex = Lock()

    # connected clients
    clients = {}
    clients_mutex = Lock()

    # class for tracking state to push out to connected clients
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

    # Handler ran on initial page load
    @app.get("/")
    def get():
        # Register a new client
        client = Client()
        with clients_mutex:
            clients[client.id] = client

        # Get current state of all checkboxes
        with checkbox_mutex:
            checkbox_array = [
                fh.CheckboxX(
                    id=f"cb-{i}", 
                    checked=val,
                    # When clicked, that checkbox will send a POST request to the server with its index
                    hx_post=f"/checkbox/toggle/{i}/{client.id}",
                ) 
                for i, val in enumerate(checkboxes)
            ]

        return (
            fh.Title('10k Checkboxes'), 
            fh.Main(
                fh.H1('Ten Thousand Checkboxes'),
                fh.Div(
                    *checkbox_array,
                    id="checkbox-array",
                ),
                cls='container',

                # use HTMX to poll for diffs to apply
                hx_trigger="every 1s", # poll every second
                hx_get=f"/diffs/{client.id}", # call the diffs endpoint
                hx_swap="none", # don't replace the entire page with returned values (more on this)
            )
        )

    # Users submitting checkbox toggles
    @app.post("/checkbox/toggle/{i}/{client_id}")
    def toggle(i: int, client_id: str):
        with checkbox_mutex:
            checkboxes[i] = not checkboxes[i]

        with clients_mutex:
            expired = []
            for client in clients.values():
                if client.id == client_id:
                    # ignore own client; it keeps its own diffs
                    continue

                # clean up old clients
                if not client.is_active(): 
                    expired.append(client.id)

                # add diff to client for when they next poll
                client.add_diff(i)
            
            for client_id in expired:
                del clients[client_id]
        return

    # Clients polling for any outstanding diffs
    @app.get("/diffs/{client_id}")
    def diffs(client_id: str):
        # we use the hx_swap_oob='true' feature to
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
                    hx_swap_oob="true", # this allows us to push updates to arbitrary checkboxes matching the id
                )
                for i in diffs
            ]
    
        return (diff_array)

    return app