# ---
# cmd: ["python", "13_sandboxes/sidecar_traffic_routing.py"]
# pytest: false
# ---

# # Filter Sandbox HTTPS traffic with a proxy Sidecar

# A proxy Sidecar can inspect and restrict HTTPS requests made by otherwise
# proxy-unaware programs in a Sandbox. This example allows requests to one
# approved Web endpoint while blocking every other destination and shows how
# the policy prevents [domain fronting](https://en.wikipedia.org/wiki/Domain_fronting)
# from inside the Sandbox.

# We use [mitmproxy](https://mitmproxy.org/) to terminate TLS, parse HTTP, connect to
# the upstream, and generate certificates. HTTPS traffic is encrypted, so to determine the
# request's destination the Sidecar must be in the middle of the connection, and perform the
# encryption/decryption against both the original client and server instead.

import tempfile
from pathlib import Path
from urllib.parse import urlparse

import modal

app = modal.App.lookup("example-sidecar-traffic-routing", create_if_missing=True)

SIDECAR_NAME = "proxy"
MITMPROXY_CONFIG_DIR = "/tmp/mitmproxy"
MITMPROXY_CA_CERT = f"{MITMPROXY_CONFIG_DIR}/mitmproxy-ca-cert.pem"
SANDBOX_CA_CERT = "/tmp/mitmproxy-ca-cert.pem"

# ## Deploy two Web endpoints to route between

# To keep the example self-contained we stand up the services the Sandbox will
# try to reach: an `allowed` endpoint the policy will permit, and a `blocked`
# endpoint standing in for a destination we do not trust. They are ordinary Modal
# [Web endpoints](https://modal.com/docs/guide/webhooks) and differ only by their
# hostname.

web_app = modal.App("example-sidecar-traffic-routing-web")
web_image = modal.Image.debian_slim().uv_pip_install("fastapi[standard]==0.139.2")


@web_app.function(image=web_image, serialized=True)
@modal.fastapi_endpoint(label="sidecar-traffic-routing-allowed")
def allowed():
    return {"endpoint": "allowed", "message": "This endpoint is on the allowlist."}


@web_app.function(image=web_image, serialized=True)
@modal.fastapi_endpoint(label="sidecar-traffic-routing-blocked")
def blocked():
    return {"endpoint": "blocked", "message": "The Sandbox should never reach this."}


with modal.enable_output():
    web_app.deploy()

allowed_url = allowed.get_web_url()
blocked_url = blocked.get_web_url()
allowed_host = urlparse(allowed_url).hostname
blocked_host = urlparse(blocked_url).hostname
print(f"Allowed endpoint: {allowed_url}")
print(f"Blocked endpoint: {blocked_url}")

# ## Define the request policy

# To configure mitmproxy dynamically we use an addon script. The `tls_clienthello`
# hook parses the hostname from the `ClientHello` [SNI](https://en.wikipedia.org/wiki/Server_Name_Indication)
# and points the upstream connection at it. After decrypting a request, the `request`
# hook applies the allowlist.

MITMPROXY_ADDON = """\
from mitmproxy import http, tls

ALLOWED_HOST = "__ALLOWED_HOST__"


def _canonical_host(name):
    return (name or "").strip().lower().rstrip(".")


def tls_clienthello(data: tls.ClientHelloData) -> None:
    hostname = _canonical_host(data.client_hello.sni)
    if hostname:
        data.context.server.address = (hostname, 443)
        data.context.server.sni = hostname


def _deny(flow: http.HTTPFlow, reason: str) -> None:
    flow.response = http.Response.make(
        403, (reason + "\\n").encode(), {"Content-Type": "text/plain"}
    )


def request(flow: http.HTTPFlow) -> None:
    sni = _canonical_host(flow.client_conn.sni)
    host = _canonical_host(flow.request.pretty_host)

    # The SNI selected the upstream, so a Host header that differs from it is a
    # domain-fronting attempt.
    if not sni or host != sni:
        _deny(flow, "Blocked: the SNI and Host header must match.")
        return

    # Default-deny allowlist: only the approved endpoint may be reached.
    if host != ALLOWED_HOST:
        _deny(flow, "Blocked: this host is not on the allowlist.")
        return
""".replace("__ALLOWED_HOST__", allowed_host)

with tempfile.TemporaryDirectory() as tmp_dir:
    addon_path = Path(tmp_dir) / "allowlist_filter.py"
    addon_path.write_text(MITMPROXY_ADDON)
    with modal.enable_output():
        sidecar_image = (
            modal.Image.debian_slim(python_version="3.12")
            .pip_install("mitmproxy==12.2.3")
            .add_local_file(addon_path, "/allowlist_filter.py", copy=True)
            .build(app)
        )

sandbox_image = modal.Image.debian_slim().apt_install("curl")

# ## Start the Sandbox and proxy Sidecar

# The experimental option names the Sidecar that receives all outbound TCP traffic on
# port 443. HTTPS fails closed until that Sidecar is running.

with modal.enable_output():
    sandbox = modal.Sandbox.create(
        "sleep",
        "600",
        app=app,
        image=sandbox_image,
        timeout=5 * 60,
        experimental_options={"proxy_traffic_via_sidecar": SIDECAR_NAME},
    )
print(f"Sandbox ID: {sandbox.object_id}")

sidecar = sandbox._experimental_sidecars.create(
    "mitmdump",
    "--mode",
    "reverse:https://invalid.invalid@443",
    "--set",
    f"confdir={MITMPROXY_CONFIG_DIR}",
    "--set",
    "connection_strategy=lazy",
    "--set",
    "keep_host_header=true",
    "--scripts",
    "/allowlist_filter.py",
    name=SIDECAR_NAME,
    image=sidecar_image,
)
print(f"Sidecar ID: {sidecar.object_id}")

# ## Trust the proxy's certificate authority

# Mitmproxy creates a unique certificate authority on first startup. Copy only its
# public certificate into the main Sandbox and pass it to curl. The CA private key
# remains isolated in the Sidecar.

read_ca = sidecar.exec(
    "bash",
    "-c",
    f"until test -s {MITMPROXY_CA_CERT} "
    "&& (echo > /dev/tcp/127.0.0.1/443) 2>/dev/null; "
    f"do sleep 0.1; done; cat {MITMPROXY_CA_CERT}",
    timeout=1 * 60,
)
ca_certificate = read_ca.stdout.read()
if read_ca.wait() != 0:
    raise RuntimeError(read_ca.stderr.read())

write_ca = sandbox.exec("tee", SANDBOX_CA_CERT)
write_ca.stdin.write(ca_certificate)
write_ca.stdin.write_eof()
write_ca.stdin.drain()
if write_ca.wait() != 0:
    raise RuntimeError(write_ca.stderr.read())

# ## Exercise the policy

# These requests use ordinary URLs with no explicit HTTP proxy settings. The first
# reaches the approved endpoint through the Sidecar; the second is answered by the
# addon with a `403` and never leaves the Sidecar.


def curl(url: str, extra_args: list[str] | None = None) -> str:
    process = sandbox.exec(
        "curl",
        "--cacert",
        SANDBOX_CA_CERT,
        "--silent",
        "--show-error",
        "--output",
        "/dev/null",
        "--write-out",
        "%{http_code}",
        *(extra_args or []),
        url,
    )
    status = process.stdout.read()
    if process.wait() != 0:
        raise RuntimeError(process.stderr.read())
    return status


print(f"GET allowed endpoint -> {curl(allowed_url)}")
print(f"GET blocked endpoint -> {curl(blocked_url)}")

# ## Block domain fronting

# [Domain fronting](https://en.wikipedia.org/wiki/Domain_fronting) hides a
# request's real destination behind an approved one. Because Modal's ingress
# routes on the HTTP `Host` header, a Sandbox can open a TLS connection with
# the *approved* endpoint's SNI while smuggling `Host: <blocked endpoint>` in
# the encrypted request. The edge would then serve the blocked endpoint.

# Our `request` hook stops this by rejecting any request whose `Host` header
# does not match the SNI. Here the Sandbox tries exactly that fronting request
# and is blocked:

fronting_status = curl(allowed_url, ["-H", f"Host: {blocked_host}"])
print(f"GET allowed SNI + blocked Host -> {fronting_status}")

# The output should look like:

# ```
# GET allowed endpoint -> 200
# GET blocked endpoint -> 403
# GET allowed SNI + blocked Host -> 403
# ```

# Terminating the Sandbox also terminates its Sidecars.

sandbox.terminate()
