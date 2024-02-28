# ---
# cmd: ["modal", "serve", "07_web_endpoints/basic_web.py"]
# ---
import modal
from modal import enter, web_endpoint

stub = modal.Stub(name="example-lifecycle-web")

# Hello world!
#
# This is as simple as it gets. A GET endpoint which
# returns a string.


@stub.function()
@web_endpoint()
def hello():
    return "Hello world!"


# Lifecycle-based.
#
# Web endpoints can be methods on a [lifecycle class](/docs/guide/lifecycle-functions#container-lifecycle-functions-and-parameters).
# This example will only set the `val` instance variable once, on container startup.
# But note that they don't need the [`modal.method`](/docs/reference/modal.method#modalmethod) decorator.


@stub.cls()
class WebApp:
    @enter()
    def startup(self):
        print("🏁 Startup up!")
        self.val = "Hello world"

    @web_endpoint()
    def web(self):
        return {"message": self.val}
