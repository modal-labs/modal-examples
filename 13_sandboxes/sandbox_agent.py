# ---
# cmd: ["python", "13_sandboxes/sandbox_agent.py"]
# pytest: false
# ---

# # Run Claude Code in a Modal Sandbox

# This example demonstrates how to run Claude Code in a Modal
# [Sandbox](https://modal.com/docs/guide/sandbox) to analyze a GitHub repository.
# The Sandbox provides an isolated environment where the agent can safely execute code
# and examine files.

import modal

app = modal.App.lookup("example-sandbox-agent", create_if_missing=True)

# First, we create a custom [Image](https://modal.com/docs/images) that has Claude Code
# and git installed.

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git")
    .env({"PATH": "/root/.local/bin:$PATH"})  # add claude to path
    .run_commands(
        "curl -fsSL https://claude.ai/install.sh | bash",
    )
)

# Then we create our Sandbox.

with modal.enable_output():
    sandbox = modal.Sandbox.create(app=app, image=image)
print(f"Sandbox ID: {sandbox.object_id}")

# Next we'll clone the repository that Claude Code will work on.
# We'll use [the Modal examples repo](https://github.com/modal-labs/modal-examples)
# that this example is a part of.

# We trigger the clone by [`exec`](https://modal.com/docs/reference/modal.Sandbox#exec)uting
# `git` as a process inside the Sandbox. We then `.wait` for it to finish.
# You can read more about the interface for managing
# `ContainerProcess`es in Sandboxes [here](https://modal.com/docs/reference/modal.container_process).

repo_url = "https://github.com/modal-labs/modal-examples"
git_ps: modal.ContainerProcess = sandbox.exec(
    "git", "clone", "--depth", "1", repo_url, "/repo"
)
git_ps.wait()
print(f"Cloned '{repo_url}' into /repo.")

# Finally we'll use `exec` again to run Claude Code to analyze the repository.
# Here, we pass the `pty` flag to give the process a
# [pseudo-terminal](https://unix.stackexchange.com/questions/21147/what-are-pseudo-terminals-pty-tty).

claude_cmd = ["claude", "-p", "What is in this repository?"]

print("\nRunning command:", *claude_cmd)

claude_ps = sandbox.exec(
    *claude_cmd,
    pty=True,  # Adding a PTY is important, since Claude requires it
    secrets=[
        modal.Secret.from_name("anthropic-secret", required_keys=["ANTHROPIC_API_KEY"])
    ],
    workdir="/repo",
)
claude_ps.wait()

# Once the command finishes, we read the `stdout` and `stderr`.

print("\nAgent stdout:\n")
print(claude_ps.stdout.read())

stderr = claude_ps.stderr.read()
if stderr != "":
    print("Agent stderr:", stderr)
