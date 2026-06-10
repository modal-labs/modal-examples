# ---
# cmd: ["python", "13_sandboxes/hermes_agent_sidecars.py", "gateway"]
# pytest: false
# lambda-test: false
# ---

# # Deploy the Hermes Agent on a Modal Sandbox with Sidecars

# [Hermes Agent](https://github.com/NousResearch/hermes-agent) is built
# around a hard split between its **"brain"** — the agent loop that does reasoning and model
# calls — and its **"hands"**: a pluggable *terminal backend* where shell and tool commands
# actually run.

# Modal [Sandbox Sidecars](https://modal.com/docs/guide/sandbox-sidecars) are built for
# exactly this shape: they let you run additional containers *alongside* a primary Sandbox, on
# the same host, on a shared internal network. So we map Hermes straight onto it:

# - The **main Sandbox container is the Hermes brain.** It runs `hermes`, holds the model API
#   key, and keeps its state (sessions, memory) on a Modal [Volume](https://modal.com/docs/guide/volumes).
# - A **Sidecar named `workspace` is the hands.** It runs `sshd`; Hermes's built-in `ssh`
#   terminal backend connects to it over the Sidecars' internal bridge network — by the Sidecar's
#   bridge IP for now (name resolution is still rolling out in the alpha) — so every shell/code tool
#   call executes there — *not* in the brain.

import argparse
import io
import secrets
import time

import modal

# ## Configuration

# Where each piece lives on the shared host.
WORKSPACE_NAME = "workspace"  # the execution Sidecar (the "hands")
SSH_PORT = 22
HERMES_HOME = "/root/.hermes"  # Hermes config + state, mounted on a Volume
SSH_KEY = "/root/.ssh/id_ed25519"  # brain's private key for reaching the workspace

# The model Hermes reasons with. With `provider: anthropic`, Hermes takes a bare model name and
# resolves it to the Anthropic API.
MODEL_PROVIDER = "anthropic"
MODEL_NAME = "claude-sonnet-4.6"

# Hermes's `config.yaml`: which model to reason with, and — crucially — `terminal.backend: ssh`,
# the line that routes every shell/tool call to the `workspace` Sidecar instead of running it in
# the brain.
HERMES_CONFIG = (
    f"model:\n  provider: {MODEL_PROVIDER}\n  default: {MODEL_NAME}\n"
    "terminal:\n  backend: ssh\n  persistent_shell: true\n"
)


# Environment that selects and configures Hermes's `ssh` terminal backend, pointed at the workspace
# Sidecar. We pass the workspace's bridge IP as the host (looked up at runtime, since Sidecar name
# resolution is still rolling out in the alpha). `accept-new` host-key handling means no
# `known_hosts` pre-seeding is needed — and since the brain's `~/.ssh` (keys and `known_hosts`)
# lives on the container filesystem rather than the Volume, a recreated workspace can't collide
# with a stale pinned host key.
def hermes_env(workspace_host: str) -> dict[str, str]:
    return {
        "TERMINAL_SSH_HOST": workspace_host,
        "TERMINAL_SSH_USER": "root",
        "TERMINAL_SSH_PORT": str(SSH_PORT),
        "TERMINAL_SSH_KEY": SSH_KEY,
        # keep one SSH shell session alive across tool calls (mirrors `persistent_shell` above)
        "TERMINAL_SSH_PERSISTENT": "true",
    }


# Hermes's web dashboard. Hermes only serves it without auth on loopback (anywhere else is
# gated behind OAuth or `--insecure`) — and loopback is exactly where we want it. The Sidecar
# network has no firewalling, so anything the brain listens on a real interface is reachable
# from the workspace, where untrusted model-generated commands run. We keep the dashboard on
# `127.0.0.1` (unreachable from the workspace, which has its own network namespace) and put an
# nginx reverse proxy with HTTP basic auth in front of it for the tunnel.
DASHBOARD_PORT = 9119  # nginx with basic auth; what the tunnel connects to
DASHBOARD_LOCAL_PORT = 9118  # `hermes dashboard`, loopback only
DASHBOARD_USER = "hermes"

MINUTES = 60  # seconds

app = modal.App.lookup("example-hermes-agent-sidecars", create_if_missing=True)

# The model key is provided via a Modal [Secret](https://modal.com/docs/guide/secrets):
anthropic_secret = modal.Secret.from_name(
    "anthropic-secret", required_keys=["ANTHROPIC_API_KEY"]
)

# Hermes keeps its state under `~/.hermes`. We back it with a Volume so sessions and memory
# survive across runs.
state_volume = modal.Volume.from_name(
    "example-hermes-agent-state", create_if_missing=True
)


# ## Images


# The **brain** image installs Hermes and the system tools it needs. We include `nodejs` because
# the gateway's web dashboard needs it, and `nginx`/`apache2-utils` to put password auth in
# front of the dashboard (see `start_dashboard` below).
brain_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ripgrep",
        "ffmpeg",
        "openssh-client",
        "nodejs",
        "npm",
        "nginx",
        "apache2-utils",
    )
    .uv_pip_install("hermes-agent==0.15.1")
    .build(app)
)


# The **workspace** image runs `sshd`. We pre-generate host keys (`ssh-keygen -A`) and harden `sshd_config` to
# accept only key-based root login; the brain's public key is injected after the Sidecar starts.
workspace_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("openssh-server", "python3", "bash", "ripgrep", "git")
    .run_commands(
        "mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh",
        "ssh-keygen -A",
        "sed -i 's/#\\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config",
        "sed -i 's/#\\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config",
        "sed -i 's/#\\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config",
    )
    .build(app)
)


# ## Helpers


# Generate an ephemeral SSH keypair *inside the brain* and authorize it on the workspace. The
# private key never leaves Modal.
def provision_ssh(sb: modal.Sandbox, workspace) -> None:
    sb.exec("ssh-keygen", "-t", "ed25519", "-N", "", "-f", SSH_KEY).wait()
    sb.exec("chmod", "600", SSH_KEY).wait()
    pubkey = sb.filesystem.read_text(f"{SSH_KEY}.pub").strip()
    workspace.filesystem.write_text(pubkey + "\n", "/root/.ssh/authorized_keys")


# Wait until the workspace's `sshd` is up *and* accepts the brain's key.
def wait_for_sshd(sb: modal.Sandbox, host: str, timeout_s: int = 60) -> None:
    deadline = time.time() + timeout_s
    while True:
        handshake = sb.exec(
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=3",
            "-i",
            SSH_KEY,
            f"root@{host}",
            "true",
        )
        if handshake.wait() == 0:
            return
        if time.time() >= deadline:
            raise TimeoutError(
                f"sshd on {host}:{SSH_PORT} not ready after {timeout_s}s:\n"
                f"{handshake.stderr.read()}"
            )
        time.sleep(2)


# Stage Hermes's config into the brain's state Volume *before* the Sandbox mounts it. The Modal SDK
# can write local files — or in-memory bytes — straight into a Volume via `batch_upload`; no running
# container required. Doing it pre-mount means the config is in place the instant the brain boots,
# with no commit/reload dance. `config.yaml` sits at the Volume root, which mounts at HERMES_HOME
# inside the brain. (`force=True` overwrites the copy left by a previous run.)
def stage_hermes_config() -> None:
    with state_volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(HERMES_CONFIG.encode()), "/config.yaml")


# Look up the workspace Sidecar's IP on the Sidecars' internal bridge network. Sidecar name
# resolution is still rolling out in the alpha, so the brain reaches the workspace by IP for now.
# `hostname -I` prints the container's non-loopback addresses; we take the first IPv4.
def container_ip(container) -> str:
    proc = container.exec("hostname", "-I")
    out, err = proc.stdout.read(), proc.stderr.read()
    ips = [tok for tok in out.split() if ":" not in tok]
    if proc.wait() != 0 or not ips:
        raise RuntimeError(f"could not determine workspace IP:\n{err}")
    return ips[0]


# nginx fronts the loopback-only dashboard with HTTP basic auth on DASHBOARD_PORT. The proxy
# rewrites `Host` to `127.0.0.1` (the dashboard rejects unexpected Host headers to defend
# against DNS rebinding) and forwards WebSocket upgrades for the in-browser chat.
NGINX_CONF = f"""
map $http_upgrade $connection_upgrade {{
    default upgrade;
    '' close;
}}
server {{
    listen {DASHBOARD_PORT};
    location / {{
        auth_basic "Hermes dashboard";
        auth_basic_user_file /etc/nginx/htpasswd;
        proxy_pass http://127.0.0.1:{DASHBOARD_LOCAL_PORT};
        proxy_set_header Host 127.0.0.1;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
    }}
}}
"""


# The Sidecar network is unfiltered in both directions (and netfilter isn't available in the
# Sandbox runtime), so a dashboard listening on a real interface would be reachable — and, since
# it manages config, sessions, and API keys, drivable — by model-generated code running in the
# workspace. Keeping the dashboard on loopback puts it out of the workspace's reach entirely,
# and the basic-auth proxy is what makes the tunnel URL alone insufficient to control the brain.
def start_dashboard(sb: modal.Sandbox, env: dict[str, str], password: str) -> None:
    htpasswd = sb.exec(
        "htpasswd", "-cbB", "/etc/nginx/htpasswd", DASHBOARD_USER, password
    )
    if htpasswd.wait() != 0:
        raise RuntimeError(f"htpasswd failed:\n{htpasswd.stderr.read()}")
    sb.filesystem.write_text(NGINX_CONF, "/etc/nginx/conf.d/dashboard.conf")
    nginx = sb.exec("nginx")
    if nginx.wait() != 0:
        raise RuntimeError(f"nginx failed to start:\n{nginx.stderr.read()}")
    sb.exec(
        "sh",
        "-c",
        f"nohup hermes dashboard --host 127.0.0.1 --port {DASHBOARD_LOCAL_PORT} "
        "--no-open >/tmp/dashboard.log 2>&1 &",
        env=env,
    )


def create_workspace(sb: modal.Sandbox):  # returns (workspace Sidecar, its bridge IP)
    try:
        workspace = sb._experimental_sidecars.create(
            "/usr/sbin/sshd",
            "-D",
            "-e",
            name=WORKSPACE_NAME,
            image=workspace_image,
        )
    except Exception as e:
        raise SystemExit(
            "Could not create a Sidecar. Sandbox Sidecars are an alpha feature: install the "
            "nightly Modal SDK (`uv pip install --prerelease=allow --upgrade modal`) and ask "
            f"Modal to enable Sidecars on your account.\n\nUnderlying error: {e}"
        )
    print(f"  workspace Sidecar: {workspace.object_id}")
    workspace_ip = container_ip(workspace)
    print(f"  workspace IP: {workspace_ip}")
    provision_ssh(sb, workspace)
    wait_for_sshd(sb, workspace_ip)
    return workspace, workspace_ip


# ## `gateway` — an always-on deployment with a web dashboard


# We boot the brain Sandbox, attach the `workspace` Sidecar, and leave Hermes running as a
# long-lived service, exposing its web dashboard over a Modal Tunnel behind HTTP basic auth
# (the password is generated fresh per deployment and printed below). Messaging platforms
# (Telegram, Discord, ...) mostly poll outbound, so they need no inbound port; add their tokens
# as Secrets and `hermes gateway run` to wire them up. We `detach()` so the Sandbox keeps running
# after this script exits — terminate it later with the `stop` command below.
def run_gateway() -> None:
    stage_hermes_config()  # into the Volume before the brain mounts it
    print("Creating the Hermes gateway Sandbox (with a tunnel for the dashboard)...")
    with modal.enable_output():
        sb = modal.Sandbox.create(
            "sleep",
            "infinity",
            app=app,
            image=brain_image,
            cpu=2.0,
            memory=4096,
            timeout=24 * 60 * MINUTES,
            idle_timeout=30 * MINUTES,  # auto-stop when idle to control cost
            volumes={HERMES_HOME: state_volume},
            secrets=[anthropic_secret],
            encrypted_ports=[
                DASHBOARD_PORT
            ],  # ingress terminates on the main Sandbox only
        )
    print(f"  gateway Sandbox: {sb.object_id}")

    _, workspace_ip = create_workspace(sb)
    env = hermes_env(workspace_ip)

    # Start the long-running gateway and the web dashboard detached, so they outlive the `exec`.
    sb.exec(
        "sh",
        "-c",
        "nohup hermes gateway run >/tmp/gateway.log 2>&1 &",
        env=env,
    )
    dashboard_password = secrets.token_urlsafe(16)
    start_dashboard(sb, env, dashboard_password)

    url = sb.tunnels()[DASHBOARD_PORT].url
    print(f"\nHermes dashboard: {url}")
    print(f"  sign in with user '{DASHBOARD_USER}', password {dashboard_password}")
    print(
        f"Leaving it running. Stop it with:\n  python {__file__.split('/')[-1]} stop {sb.object_id}"
    )
    sb.detach()


def stop(sandbox_id: str) -> None:
    modal.Sandbox.from_id(sandbox_id).terminate()
    print(f"Terminated {sandbox_id}.")


# ## Run it
#
# - `python 13_sandboxes/hermes_agent_sidecars.py gateway` — deploy the always-on dashboard.
# - `python 13_sandboxes/hermes_agent_sidecars.py stop <sandbox-id>` — stop a gateway.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hermes Agent on Modal Sandbox + Sidecars"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "gateway", help="deploy an always-on gateway with a dashboard tunnel"
    )
    stop_parser = sub.add_parser("stop", help="terminate a gateway Sandbox by id")
    stop_parser.add_argument("sandbox_id")

    args = parser.parse_args()
    if args.command == "gateway":
        run_gateway()
    elif args.command == "stop":
        stop(args.sandbox_id)


if __name__ == "__main__":
    main()
