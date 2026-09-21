# ---
# cmd: ["python", "13_sandboxes/sandbox_secret_injection.py"]
# pytest: false
# lambda-test: false
# ---

# # Inject secrets into Sandbox HTTPS requests with an Outbound Policy

# [Sandboxes](https://modal.com/docs/guide/sandboxes) are often used for agents
# which will run untrusted code. The Sandbox provides a great security boundary
# to contain the untrusted code, but an agent may need to access some external
# API over network calls to do useful work. If the API requires an API token
# for authorization we may be reluctant to give it to the agent running in the
# Sandbox, as it might decide to exfiltrate it and send it to someone who
# shouldn't have access to the token.

# An [Outbound Policy](https://modal.com/docs/guide/sandbox-secret-injection)
# solves this by injecting headers into outbound HTTPS requests _outside_ the
# Sandbox. The secret never appears in the Sandbox's environment, but we can
# still make authorized calls to our API.

# This example stands in for the external API with a mock weather API deployed
# as a Modal [Web Function](https://modal.com/docs/guide/webhooks). It requires
# an `Authorization: Bearer <key>` header, which the Sandbox workload never
# sees.

import hmac
import os
import textwrap
from urllib.parse import urlparse

import fastapi
import modal

app = modal.App.lookup("example-sandbox-outbound-policy", create_if_missing=True)

MINUTES = 60  # seconds

# ## Deploy a mock authenticated API

# As a stand-in for an external authenticated API, we deploy a small Web
# Function that checks the `Authorization` header against a value stored in a
# Modal [Secret](https://modal.com/docs/guide/secrets).

# Note that this is not a tutorial on web security. This is not a production
# grade authorization scheme, please don't treat it as such.

SECRET_NAME = "example-outbound-policy-api-key"
API_KEY = "hunter2"
modal.Secret.objects.create(SECRET_NAME, {"API_KEY": API_KEY}, allow_existing=True)
api_secret = modal.Secret.from_name(SECRET_NAME, required_keys=["API_KEY"])


web_app = modal.App("example-outbound-policy-api")
web_image = modal.Image.debian_slim().uv_pip_install("fastapi[standard]==0.139.2")


@web_app.function(image=web_image, secrets=[api_secret], serialized=True)
@modal.fastapi_endpoint()
def weather(request: fastapi.Request):
    expected = os.environ["API_KEY"]
    auth = request.headers.get("authorization", "")
    if not hmac.compare_digest(auth, f"Bearer {expected}"):
        raise fastapi.HTTPException(status_code=401, detail="unauthorized")

    response = {
        "city": request.query_params.get("city", "Tokyo"),
        "temperature_c": 21,
        "conditions": "sunny",
    }
    return response


with modal.enable_output():
    web_app.deploy()
    image = modal.Image.debian_slim().build(app)

api_url = weather.get_web_url()
if not api_url:
    raise RuntimeError("expected a web URL for the weather endpoint")
api_host = urlparse(api_url).hostname

# ## Define the Outbound Policy

# We can define an Outbound Policy with a header replacement that uses the
# Secret with the API key and injects it into outgoing HTTPS requests from a
# Sandbox. We scope the replacement to a single domain so we don't leak the
# secret to other services that might be called from within the Sandbox.

outbound_policy = modal.OutboundPolicy().with_header_replacement(
    domain=api_host,
    secret=api_secret,
    headers={"Authorization": "Bearer $API_KEY"},
)

# ## Create the Sandbox and call the API

# We create a Sandbox which uses the Outbound Policy. We can call the API using
# Python from the Sandbox to make a plain HTTPS request. Note that we do not
# attach any headers at all here.

sb = modal.Sandbox.create(
    "sleep",
    str(5 * MINUTES),
    app=app,
    image=image,
    outbound_policy=outbound_policy,
)


def call_api() -> tuple[str, str]:
    script = textwrap.dedent(f"""
        import http.client

        client = http.client.HTTPSConnection({api_host!r})
        client.request('GET', '/?city=Tokyo')
        response = client.getresponse()
        print(response.status)
        print(response.read().decode())
    """)
    p = sb.exec("python", "-c", script)
    p.wait()
    status, _, body = p.stdout.read().partition("\n")
    return status.strip(), body


status, body = call_api()
print(body)

# We assert that the API accepted the call. Without the Outbound Policy this
# would've returned a 401.

assert status == "200", body
assert "temperature_c" in body

# Meanwhile the key is nowhere in the Sandbox's environment:

p = sb.exec("printenv")
p.wait()
assert API_KEY not in p.stdout.read()

# ## Clean up

sb.terminate()
