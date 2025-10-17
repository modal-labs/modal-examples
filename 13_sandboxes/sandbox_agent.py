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

# First we create a custom Image that has Claude Code installed.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git")
    .run_commands(
        "curl -fsSL https://claude.ai/install.sh | bash",
    )
    .env({"PATH": "/root/.local/bin:$PATH"})  # add claude to path
)

with modal.enable_output():
    sandbox = modal.Sandbox.create(app=app, image=image)
print(f"Sandbox ID: {sandbox.object_id}")

# Next we'll clone a repository that Claude Code will work on.
repo_url = "https://github.com/modal-labs/modal-examples"
git_ps = sandbox.exec("git", "clone", repo_url, "/repo")
git_ps.wait()
print(f"Cloned '{repo_url}' into /repo.")

# Finally we'll run Claude Code to analyze the repository.
claude_cmd = ["claude", "-p", "What is in this repository?"]

print("\nRunning command:", claude_cmd)

claude_ps = sandbox.exec(
    *claude_cmd,
    pty=True,  # Adding a PTY is important, since Claude requires it
    secrets=[
        modal.Secret.from_name("anthropic-secret", required_keys=["ANTHROPIC_API_KEY"])
    ],
    workdir="/repo",
)
claude_ps.wait()

print("\nAgent stdout:\n")
print(claude_ps.stdout.read())

stderr = claude_ps.stderr.read()
if stderr != "":
    print("Agent stderr:", stderr)
