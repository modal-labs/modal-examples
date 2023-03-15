# ---
# integration-test: false
# lambda-test: false
# ---

"""Example of setting up tailscale VPN (https://tailscale.com/) as a sidecar

In this example, it's used to access a private API in another machine on the tailnet
The `tailscale_sidecar` utility function contains most of the setup, and needs to
be called from the container defined in `tailscale_image`.
"""

import contextlib
import os
import subprocess
import time

import modal

TAILSCALE_DOWNLOAD = "tailscale_1.36.0_amd64.tgz"
tailscale_image = (
    modal.Image.debian_slim()
    .apt_install("wget")
    .dockerfile_commands(
        [
            "WORKDIR /tailscale",
            f"RUN wget https://pkgs.tailscale.com/stable/{TAILSCALE_DOWNLOAD} && \
tar xzf {TAILSCALE_DOWNLOAD} --strip-components=1",
            "RUN mkdir -p /tmp/tailscale",
        ]
    )
    .pip_install("requests[socks]")
)

stub = modal.Stub(image=tailscale_image)


@contextlib.contextmanager
def tailscale_sidecar(tailscale_authkey, show_output=False):
    """Context manager that sets up a tailscale userspace sidecar

    Enables both SOCKS5 and HTTP proxies on localhost:1055 and sets
    environment variables ALL_PROXY, HTTP_PROXY and http_proxy accordingly
    """
    PROXY_SIDECAR_CMD = [
        "/tailscale/tailscaled",
        "--tun=userspace-networking",
        "--socks5-server=localhost:1055",
        "--outbound-http-proxy-listen=localhost:1055",
    ]
    AUTH_CMD = [
        "/tailscale/tailscale",
        "up",
        f"--authkey={tailscale_authkey}",
        "--hostname=modal-app",
    ]

    if not show_output:
        output_args = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        output_args = {}
    with subprocess.Popen(PROXY_SIDECAR_CMD, **output_args) as p:  # type: ignore
        subprocess.check_call(AUTH_CMD)
        # wait for tailscale to fully configure network, otherwise proxies can fail:
        time.sleep(2)
        os.environ["ALL_PROXY"] = "socks5://localhost:1055/"
        for key in ["HTTP_PROXY", "http_proxy"]:
            os.environ[key] = "http://localhost:1055/"
        yield
        subprocess.check_call(
            ["/tailscale/tailscale", "logout"]
        )  # removes node from tailnet
        p.kill()  # stop sidecar daemon


@stub.function(secrets=[modal.Secret.from_name("tailscale-auth")])
def tail():
    import requests

    TAILSCALE_AUTHKEY = os.environ["TAILSCALE_AUTHKEY"]
    with tailscale_sidecar(TAILSCALE_AUTHKEY, show_output=False):
        resp = requests.get("http://raspberrypi:5000")
        print(resp.content)
