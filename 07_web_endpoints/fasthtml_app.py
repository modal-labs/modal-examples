# ---
# cmd: ["modal", "serve", "07_web_endpoints/fasthtml_app.py"]
# ---

# # Deploy a FastHTML app with Modal

# This example shows how you can deploy a FastHTML app with Modal.
# [FastHTML](https://www.fastht.ml/) is a Python library built on top of [HTMX](https://htmx.org/)
# which allows you to create entire web applications using only Python.

# The integration is pretty simple, thanks to the ASGI standard.
# You just need to define a function returns your FastHTML app
# and is decorated with `app.function` and `modal.asgi_app`.

import modal

app = modal.App("example-fasthtml")


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install(
        "python-fasthtml==0.5.2"
    )
)
@modal.asgi_app()
def serve():
    import fasthtml.common as fh

    app = fh.FastHTML()

    @app.get("/")
    def home():
        return fh.Div(fh.P("Hello World!"), hx_get="/change")

    return app
