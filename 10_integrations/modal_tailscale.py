"""Example of setting up tailscale VPN (https://tailscale.com/) as a sidecar

In this example, it's used to access a private API in another machine on the tailnet
The `tailscale_sidecar` utility function contains most of the setup, and needs to
be called from the container defined in `tailscale_image`.
"""

import contextlib
import os
import subprocess
import sys
import threading

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
    working_tailscale = threading.Event()

    def output_watcher(tailscaled_output):
        for line in tailscaled_output:
            if show_output:
                print(f"TAILSCALE: {line}", file=sys.stderr)
            if b"magicsock: derp-1 connected" in line:
                working_tailscale.set()

    with subprocess.Popen(
        PROXY_SIDECAR_CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) as p:
        subprocess.check_call(AUTH_CMD)
        # wait for tailscale to fully configure userspace, otherwise proxies can fail:
        t = threading.Thread(
            target=output_watcher, args=(p.stdout,), daemon=True
        )
        t.start()
        working_tailscale.wait()
        os.environ["ALL_PROXY"] = "socks5://localhost:1055/"
        for key in ["HTTP_PROXY", "http_proxy"]:
            os.environ[key] = "http://localhost:1055/"
        yield
        subprocess.check_call(
            ["/tailscale/tailscale", "logout"]
        )  # removes node from tailnet
        p.kill()  # stop sidecar daemon


tailscale_secret = modal.Secret.from_name("tailscale-auth")


@stub.function(secrets=[tailscale_secret])
def tail():
    import requests

    TAILSCALE_AUTHKEY = os.environ["TAILSCALE_AUTHKEY"]
    with tailscale_sidecar(TAILSCALE_AUTHKEY):
        resp = requests.get("http://raspberrypi:5000")
        print(resp.content)
