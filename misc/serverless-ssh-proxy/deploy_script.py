import os

import modal

app = modal.App("ssh-proxy-app")


ssh_proxy_dir = os.path.dirname(os.path.abspath(__file__)) + "/ssh-proxy"

ssh_proxy_image = (
    modal.Image.debian_slim()
    .apt_install("curl", "pkg-config", "libssl-dev", "protobuf-compiler")
    .run_commands("curl https://sh.rustup.rs -sSf | sh -s -- -y")
    .add_local_dir(ssh_proxy_dir, remote_path="/ssh-proxy")
)


modal_secrets = {
    "TOKEN_ID": os.environ.get("MODAL_TOKEN_ID"),
    "TOKEN_SECRET": os.environ.get("MODAL_TOKEN_SECRET"),
    "SSH_PUBLIC_KEY": os.environ.get("SSH_PUBLIC_KEY"),
    "RUSTFLAGS": " -Awarnings",
    "MODAL_ENVIRONMENT_NAME": os.environ.get("MODAL_ENVIRONMENT_NAME", "main"),
}


@app.function(
    secrets=[modal.Secret.from_dict(modal_secrets)],
    image=ssh_proxy_image,
    timeout=30 * 60,
)
def run_ssh_proxy():
    import subprocess

    subprocess.run(["ls", "/ssh-proxy"], check=True)
    with modal.forward(22, unencrypted=True) as tunnel:
        (hostname, port) = tunnel.tcp_socket

        print(f"""Tunnel created\n. Use the following SSH command to connect:
        ssh -p {port} -i [private_key_path] user@{hostname}
        Note that in this implementation user can be any string.""")
        result = subprocess.run(
            ["/root/.cargo/bin/cargo", "run", "--release", "--bin", "ssh_proxy"],
            check=True,
            cwd="/ssh-proxy",
        )
    return result.returncode


if __name__ == "__main__":
    run_ssh_proxy.remote()
