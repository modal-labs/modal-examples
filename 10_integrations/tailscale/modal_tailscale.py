# ---
# lambda-test: false
# ---

# # Add Modal Apps to Tailscale
#
# This example demonstrates how to integrate Modal with Tailscale (https://tailscale.com).
# It outlines the steps to configure Modal containers so that they join the Tailscale network.

# We use a custom entrypoint to automatically add containers to a Tailscale network (tailnet).
# This configuration enables the containers to interact with one another and with
# additional applications within the same tailnet.


import modal

# Install Tailscale and copy custom entrypoint script ([entrypoint.sh](https://github.com/modal-labs/modal-examples/blob/main/10_integrations/tailscale/entrypoint.sh)). The script must be
# executable.
image = (
    modal.Image.debian_slim()
    .apt_install("curl")
    .run_commands("curl -fsSL https://tailscale.com/install.sh | sh")
    .pip_install("requests[socks]")
    .copy_local_file("./entrypoint.sh", "/root/entrypoint.sh")
    .dockerfile_commands(
        "RUN chmod a+x /root/entrypoint.sh",
        'ENTRYPOINT ["/root/entrypoint.sh"]',
    )
)
app = modal.App(image=image)


# Run your function adding a Tailscale secret. It expects an environment variable
# named `TAILSCALE_AUTHKEY`. We suggest creating a [reusable and ephemeral key](https://tailscale.com/kb/1111/ephemeral-nodes).
@app.function(
    secrets=[
        modal.Secret.from_name("tailscale-auth"),
        modal.Secret.from_dict(
            {
                "ALL_PROXY": "socks5://localhost:1080/",
                "HTTP_PROXY": "http://localhost:1080/",
                "http_proxy": "http://localhost:1080/",
            }
        ),
    ],
)
def connect_to_raspberrypi():
    import requests

    # Connect to other machines in your tailnet.
    resp = requests.get("http://raspberrypi:5000")
    print(resp.content)


# Run this script with `modal run modal_tailscale.py`. You will see Tailscale logs
# when the container start indicating that you were able to login successfully and
# that the proxies (SOCKS5 and HTTP) have created been successfully. You will also
# be able to see Modal containers in your Tailscale dashboard in the "Machines" tab.
# Every new container launched will show up as a new "machine". Containers are
# individually addressable using their Tailscale name or IP address.
