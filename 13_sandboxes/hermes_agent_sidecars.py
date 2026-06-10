# ---
# cmd: ["modal", "run", "13_sandboxes/hermes_agent_sidecars.py"]
# pytest: false
# lambda-test: false
# ---

# # Deploy a Hermes agent with its tools in a Sidecar

# [Hermes Agent](https://github.com/nousresearch/hermes-agent) is Nous Research's
# open-source autonomous agent: a model loop with persistent memory, self-authored
# skills, and tools for running shell commands and editing files. Like any agent,
# it has a "brain" (the loop that calls an LLM API, holding an API key) and "hands"
# (the tool calls the model asks for - arbitrary, model-generated shell commands).

# In this example we run Hermes in a Modal
# [Sandbox](https://modal.com/docs/guide/sandbox) and use
# [Sandbox Sidecars](https://modal.com/docs/guide/sandbox-sidecars) to split the
# brain and hands across containers:

# - The **main container** runs the Hermes agent loop and holds `ANTHROPIC_API_KEY`
#   as a [Modal Secret](https://modal.com/docs/guide/secrets).
# - A **Sidecar** named `tools` runs nothing but an SSH server - and gets **no
#   Secrets at all**. Hermes's built-in SSH terminal backend routes every
#   model-driven tool call (shell *and* file operations) into it over the
#   Sandbox's internal bridge network.

# `modal deploy` this file and you get a personal agent with two surfaces:

# 1. An `ask` [Function](https://modal.com/docs/guide/apps) you can call from
#    anywhere.
# 2. The Hermes web dashboard behind an authenticated
#    [Tunnel](https://modal.com/docs/guide/tunnels).

# Agent state (sessions, memories, skills) persists in a Modal
# [Volume](https://modal.com/docs/guide/volumes), so your agent remembers you
# across Sandbox restarts.

# ## Setup

# The only required Secret holds your Anthropic API key:

# ```bash
# modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...
# ```

# A second Secret protects the web dashboard. Hermes refuses to bind the
# dashboard to a non-loopback address without an auth provider, so we configure
# its built-in basic auth:

# ```bash
# modal secret create hermes-dashboard-secret \
#     HERMES_DASHBOARD_BASIC_AUTH_USERNAME=hermes \
#     HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=... \
#     HERMES_DASHBOARD_BASIC_AUTH_SECRET=...  # 32+ random bytes, signs sessions
# ```

import modal

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

APP_NAME = "example-hermes-agent-sidecars"
SANDBOX_APP_NAME = f"{APP_NAME}-sandboxes"
SANDBOX_NAME = "hermes-agent"

HERMES_VERSION = "0.16.0"
MODEL = "claude-opus-4-8"  # any Anthropic model name works here
TOOLS_NAME = "tools"  # the Sidecar's name doubles as its hostname
SSH_KEY = "/root/.ssh/id_ed25519"

DASHBOARD_PORT = 9119
STATE_MOUNT_POINT = "/state"  # the Volume mount point: durable copies of agent state
HERMES_HOME = "/root/.hermes"  # local disk: live agent state
READY_MARKER = "/root/.hermes-ready"

# The deployed App holds our control Functions. The Sandbox itself lives in a
# *separate* App, looked up by name at runtime - so a quick `modal run` (which
# uses an ephemeral App) and a durable `modal deploy` both talk to the same
# long-lived agent, and redeploys never touch it.

app = modal.App(APP_NAME)

state_volume = modal.Volume.from_name(f"{APP_NAME}-state", create_if_missing=True)

# ## Two Images, two trust levels

# The main container's Image installs Hermes from PyPI, with the extras for the
# web dashboard. Node is there for the dashboard's web UI build on first launch.

agent_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "openssh-client", "rsync", "nodejs", "npm")
    .uv_pip_install(f"hermes-agent[web,pty]=={HERMES_VERSION}")
)

# The `tools` Sidecar's Image is deliberately boring: an SSH server (keys only,
# no passwords) and whatever you want your agent's hands to be able to use.

tools_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("openssh-server", "git", "curl", "ripgrep")
    .run_commands(
        "mkdir -p /run/sshd /workspace",
        "ssh-keygen -A",
        "printf 'PermitRootLogin prohibit-password\\nPasswordAuthentication no\\n'"
        " > /etc/ssh/sshd_config.d/hermes.conf",
    )
)

# ## Pointing Hermes at the Sidecar

# Hermes's terminal backend is selected and configured entirely through
# environment variables, which we set once on the Sandbox so that every process
# in it (each `hermes -z` one-shot, and the dashboard) inherits them.
# `TERMINAL_ENV=ssh` makes every `terminal` tool call run over SSH on the
# Sidecar, and Hermes's file tools (read/write/patch/search) follow the same
# backend so they execute there too.

TERMINAL_ENV = {
    "TERMINAL_ENV": "ssh",
    "TERMINAL_SSH_HOST": TOOLS_NAME,
    "TERMINAL_SSH_USER": "root",
    "TERMINAL_SSH_KEY": SSH_KEY,
    "TERMINAL_CWD": "/workspace",
}

# The rest of the configuration goes in `$HERMES_HOME/config.yaml`: the model,
# and a restricted toolset. `terminal` and `file` route to the Sidecar;
# everything that executes model-generated *commands* therefore runs in the
# secret-free container. The `memory` tool - fixed Hermes code that only reads
# and writes the agent's own notes under `HERMES_HOME` - stays in the main
# container, where state persists to the Volume.

CONFIG_YAML = f"""\
model:
  default: "anthropic/{MODEL}"
  provider: "anthropic"

platform_toolsets:
  cli: [terminal, file, memory]
"""

# ## Small helpers

# Run a command in a container (the Sandbox or a Sidecar), returning its exit
# code, stdout, and stderr. (`SidecarContainer` isn't exported at the top of
# `modal` yet, only as `modal.sandbox.SidecarContainer`.)


def run(
    container: modal.Sandbox | modal.sandbox.SidecarContainer,
    *cmd: str,
) -> tuple[int, str, str]:
    proc = container.exec(*cmd)
    stdout, stderr = proc.stdout.read(), proc.stderr.read()
    return proc.wait(), stdout, stderr


# Retry a shell test once per second until it passes.


def wait_for(
    container: modal.Sandbox | modal.sandbox.SidecarContainer,
    shell_test: str,
    attempts: int,
    what: str,
) -> None:
    code, _, stderr = run(
        container,
        "sh",
        "-c",
        f"for i in $(seq {attempts}); do {shell_test} && exit 0; sleep 1; done; exit 1",
    )
    if code != 0:
        raise RuntimeError(f"timed out waiting for {what}: {stderr.strip()}")


# ## One long-lived agent Sandbox

# There is exactly one agent, so we use a *named* Sandbox: `Sandbox.from_name`
# finds it if it's running, and creating it with the same name from two racing
# calls makes the loser raise, fall back to the winner's Sandbox, and wait for
# bootstrap to finish (signalled by `READY_MARKER`).


def get_or_create_sandbox() -> modal.Sandbox:
    try:
        sb = modal.Sandbox.from_name(SANDBOX_APP_NAME, SANDBOX_NAME)
        wait_for(sb, f"test -f {READY_MARKER}", 10 * MINUTES, "agent bootstrap")
        return sb
    except modal.exception.NotFoundError:
        pass

    sandbox_app = modal.App.lookup(SANDBOX_APP_NAME, create_if_missing=True)

    print("Building Images...")
    built_agent_image = agent_image.build(sandbox_app)
    built_tools_image = tools_image.build(sandbox_app)

    secrets = [
        modal.Secret.from_name("anthropic-secret", required_keys=["ANTHROPIC_API_KEY"]),
        modal.Secret.from_name(
            "hermes-dashboard-secret",
            required_keys=[
                "HERMES_DASHBOARD_BASIC_AUTH_USERNAME",
                "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD",
            ],
        ),
    ]

    # The main container gets no command: it idles, waiting for our `exec`s. We
    # use a plain `timeout` rather than `idle_timeout` - the daemons inside
    # never go idle - and recreate on demand after it expires; agent state
    # survives in the Volume. (Note that the Sidecar shares this Sandbox's CPU
    # and memory allocation, so we size for both.)

    try:
        sb = modal.Sandbox.create(
            app=sandbox_app,
            name=SANDBOX_NAME,
            image=built_agent_image,
            env=TERMINAL_ENV,
            secrets=secrets,
            volumes={STATE_MOUNT_POINT: state_volume},
            encrypted_ports=[DASHBOARD_PORT],
            cpu=2,
            memory=4096,
            timeout=24 * HOURS,
        )
    except modal.exception.AlreadyExistsError:  # we lost a creation race
        sb = modal.Sandbox.from_name(SANDBOX_APP_NAME, SANDBOX_NAME)
        wait_for(sb, f"test -f {READY_MARKER}", 10 * MINUTES, "agent bootstrap")
        return sb

    try:
        bootstrap(sb, built_tools_image)
    except BaseException:
        sb.terminate()  # don't leave a half-bootstrapped agent behind
        raise
    return sb


# ## Bootstrap: Sidecar, SSH, state, daemons


def bootstrap(sb: modal.Sandbox, tools_image: modal.Image) -> None:
    # First, attach the Sidecar. Its only process is `sshd`, and gets no `secrets`.

    tools = sb._experimental_sidecars.create(
        "/usr/sbin/sshd",
        "-D",
        "-e",
        name=TOOLS_NAME,
        image=tools_image,
        workdir="/workspace",
    )
    print(f"tools Sidecar: {tools.object_id}")

    # Generate an SSH keypair in the main container and authorize it in the
    # Sidecar, then probe until sshd answers. The Sidecar is resolvable as
    # `tools` on the bridge network. The probe also records the host key
    # (`accept-new`), so Hermes's later connections are warning-free.

    run(
        sb,
        "sh",
        "-c",
        f"mkdir -p /root/.ssh && ssh-keygen -q -t ed25519 -N '' -f {SSH_KEY}",
    )
    pubkey = sb.filesystem.read_text(f"{SSH_KEY}.pub")
    tools.filesystem.write_text(pubkey, "/root/.ssh/authorized_keys")
    run(
        tools,
        "sh",
        "-c",
        "chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys",
    )
    ssh = (
        "ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
        f" -o ConnectTimeout=2 -i {SSH_KEY} root@{TOOLS_NAME}"
    )
    wait_for(sb, f"{ssh} true", 60, "sshd in the tools Sidecar")

    # ### Persisting state

    # Hermes keeps sessions and memory in a SQLite database, which needs file
    # locking that the network-backed Volume mount doesn't provide. So live
    # state stays on the container's local disk: we restore it from the Volume
    # at boot and snapshot it back with a background loop every minute. An
    # abrupt kill can lose up to a minute of session state. (The excludes skip
    # toolchains Hermes may bootstrap into its home directory.)

    sync = "rsync -a --exclude=node --exclude=bin"
    run(
        sb,
        "sh",
        "-c",
        f"mkdir -p {HERMES_HOME} {STATE_MOUNT_POINT}/hermes"
        f" && {sync} {STATE_MOUNT_POINT}/hermes/ {HERMES_HOME}/",
    )
    sb.filesystem.write_text(CONFIG_YAML, f"{HERMES_HOME}/config.yaml")
    sb.exec(
        "sh",
        "-c",
        f"while true; do sleep 60; {sync} {HERMES_HOME}/ {STATE_MOUNT_POINT}/hermes/; done"
        " >>/var/log/state-sync.log 2>&1",
    )

    # Long-running daemons redirect their own output to files *inside* the
    # Sandbox: the processes belong to the Sandbox, not to our local handles,
    # and the logs are there when you `modal shell` in to debug.

    sb.exec(
        "sh",
        "-c",
        f"hermes dashboard --host 0.0.0.0 --port {DASHBOARD_PORT} --no-open"
        " >>/var/log/dashboard.log 2>&1",
    )
    # The first launch builds the dashboard's web UI, so be patient.
    wait_for(
        sb,
        f"curl -s -o /dev/null http://127.0.0.1:{DASHBOARD_PORT}/",
        10 * MINUTES,
        "the Hermes dashboard",
    )

    sb.filesystem.write_text("ok\n", READY_MARKER)
    print(f"agent Sandbox ready: {sb.object_id}")


# ## Talk to the agent

# `ask` is the programmatic surface: a headless one-shot conversation turn
# (`hermes -z`). The API key reaches the Hermes process via the Sandbox's
# Secret, while the prompt's tool calls run in the Sidecar.


@app.function(timeout=30 * MINUTES)
def ask(prompt: str) -> str:
    sb = get_or_create_sandbox()
    proc = sb.exec("hermes", "-z", prompt)
    stdout, stderr = proc.stdout.read(), proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(f"hermes exited non-zero:\n{stderr}")
    return stdout


# The dashboard is reachable through a Modal Tunnel. The URL changes if the
# Sandbox is recreated, so we expose it as a Function instead of printing it
# once and hoping.


@app.function(timeout=30 * MINUTES)
def dashboard_url() -> str:
    sb = get_or_create_sandbox()
    return sb.tunnels()[DASHBOARD_PORT].url


# ## Try it

# Run a one-shot prompt (the default one demonstrates the isolation: the tool
# call lands in the Sidecar, where there is no API key):

# ```bash
# modal run 13_sandboxes/hermes_agent_sidecars.py
# modal run 13_sandboxes/hermes_agent_sidecars.py --prompt "..."
# ```

# Then make it permanent with `modal deploy 13_sandboxes/hermes_agent_sidecars.py`
# and call it from anywhere:

# ```python
# modal.Function.from_name("example-hermes-agent-sidecars", "ask").remote("...")
# ```


@app.local_entrypoint()
def main(
    prompt: str = (
        "Use your terminal tool to run `hostname -I` and"
        " `printenv`, and report exactly what you see."
    ),
):
    print(ask.remote(prompt))
    print("\nDashboard:", dashboard_url.remote())
