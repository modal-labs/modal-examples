"""Coding agent backends for sandbox-based code generation.

Supports Claude Code CLI and OpenAI (via Responses API) as configurable
coding agents. Each agent runs inside a Modal Sandbox with docs mounted
as files, providing isolated environments with controlled documentation access.
"""

import re
import socket
import time
from dataclasses import dataclass, field

import modal

# --- Sandbox Images ---

claude_code_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git")
    .env({"PATH": "/root/.local/bin:$PATH"})
    .run_commands("curl -fsSL https://claude.ai/install.sh | bash")
)

codex_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git")
    .pip_install("openai")
)

AGENT_IMAGES = {
    "claude": claude_code_image,
    "codex": codex_image,
}

AGENT_SECRETS = {
    "claude": "anthropic-secret",
    "codex": "openai-secret",
}

# --- Network Isolation ---

# API hostnames that coding agents need to reach.
# All other outbound traffic is blocked when network isolation is enabled.
AGENT_API_HOSTS = {
    "claude": ["api.anthropic.com"],
    "codex": ["api.openai.com"],
}


def resolve_cidr_allowlist(agent_type: str) -> list[str]:
    """Resolve LLM API hostnames to /24 CIDR ranges for the sandbox allowlist.

    Uses DNS resolution at call time to get current IPs, then broadens
    to /24 ranges to handle CDN IP rotation within the same subnet.
    """
    cidrs = set()
    hosts = AGENT_API_HOSTS.get(agent_type, [])
    for host in hosts:
        try:
            # getaddrinfo returns all resolved addresses (IPv4 and IPv6)
            results = socket.getaddrinfo(host, 443, socket.AF_INET)
            for _, _, _, _, sockaddr in results:
                ip = sockaddr[0]
                # Use /24 to handle CDN IP rotation within same subnet
                prefix = ".".join(ip.split(".")[:3])
                cidrs.add(f"{prefix}.0/24")
        except socket.gaierror:
            pass  # DNS resolution failed; sandbox will fail if no IPs resolved
    return sorted(cidrs)


# --- Prompts ---

AGENT_PROMPT = """\
Read the documentation file at /workspace/docs.txt carefully. \
Based ONLY on that documentation, write a complete, runnable Python program \
for the following task:

{task_description}

Requirements:
- Write complete Python code that can be run with `modal run script.py`
- Include all necessary imports
- Use ONLY the APIs and patterns described in the documentation
- Do NOT use any prior knowledge that contradicts the documentation

Save your final solution to /workspace/solution.py"""

# Python script that runs inside the sandbox to call the OpenAI API.
# This replaces the Codex CLI (which requires OAuth login) with a direct
# API call using OPENAI_API_KEY env var.
_OPENAI_AGENT_SCRIPT = r"""
import re
import sys

from openai import OpenAI

model = sys.argv[1]
prompt = sys.argv[2]

client = OpenAI()  # reads OPENAI_API_KEY from env

response = client.responses.create(
    model=model,
    instructions="You are a coding assistant. Write code to files as requested.",
    input=prompt,
)

# Extract the text output
text = ""
for item in response.output:
    if item.type == "message":
        for block in item.content:
            if hasattr(block, "text"):
                text += block.text

# Try to extract code and write solution.py
patterns = [
    r"```python\s*\n(.*?)```",
    r"```\s*\n(.*?)```",
]
code = None
for pattern in patterns:
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        break

if code:
    with open("/workspace/solution.py", "w") as f:
        f.write(code)
    print(code)
else:
    # The response might be raw code without markdown fences
    with open("/workspace/solution.py", "w") as f:
        f.write(text)
    print(text)
"""


@dataclass
class AgentConfig:
    """Configuration for a coding agent."""

    agent_type: str  # "claude" or "codex"
    model: str | None = None
    timeout: int = 300  # Sandbox timeout in seconds
    network_isolated: bool = True  # Restrict network to only LLM API endpoints
    extra_params: dict = field(default_factory=dict)

    @property
    def resolved_model(self) -> str:
        if self.model:
            return self.model
        defaults = {
            "claude": "claude-sonnet-4-20250514",
            "codex": "o3-mini",
        }
        return defaults.get(self.agent_type, self.agent_type)


def extract_code(text: str) -> str:
    """Extract Python code from text output (fallback if no solution file)."""
    # Try to find code in ```python blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Try to find code in generic ``` blocks
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    return text.strip()


def run_agent_in_sandbox(
    task_description: str,
    docs_content: str,
    config: AgentConfig,
    app: modal.App,
) -> tuple[str, float]:
    """Run a coding agent in a Modal Sandbox and return (generated_code, elapsed_time).

    Creates a sandbox with the agent CLI installed, writes docs as files,
    runs the agent with a task prompt, and extracts the generated code.
    """
    image = AGENT_IMAGES.get(config.agent_type)
    if image is None:
        raise ValueError(
            f"Unknown agent type: {config.agent_type}. "
            f"Available: {list(AGENT_IMAGES.keys())}"
        )

    secret_name = AGENT_SECRETS[config.agent_type]
    secrets = [modal.Secret.from_name(secret_name)]

    # Resolve CIDR allowlist for LLM API endpoints
    sandbox_kwargs: dict = {
        "app": app,
        "image": image,
        "timeout": config.timeout,
    }
    if config.network_isolated:
        cidr_allowlist = resolve_cidr_allowlist(config.agent_type)
        if not cidr_allowlist:
            raise RuntimeError(
                f"Failed to resolve API IPs for {config.agent_type}. "
                "Cannot create network-isolated sandbox without allowlist."
            )
        # cidr_allowlist restricts outbound traffic to only the listed CIDRs.
        # This blocks access to modal.com docs while allowing LLM API calls.
        sandbox_kwargs["cidr_allowlist"] = cidr_allowlist

    # Create sandbox (long-running, we exec commands into it)
    sandbox = modal.Sandbox.create(**sandbox_kwargs)

    start_time = time.time()

    try:
        # Create workspace and write docs
        sandbox.mkdir("/workspace", parents=True)
        f = sandbox.open("/workspace/docs.txt", "w")
        f.write(docs_content)
        f.close()

        # Initialize a git repo in /workspace (required by Claude Code CLI)
        if config.agent_type == "claude":
            for git_cmd in [
                ["git", "init"],
                ["git", "config", "user.email", "eval@modal.com"],
                ["git", "config", "user.name", "Eval"],
                ["git", "add", "."],
                ["git", "commit", "-m", "init", "--allow-empty"],
            ]:
                git_ps = sandbox.exec(*git_cmd, workdir="/workspace")
                git_ps.wait()

        # Build prompt
        prompt = AGENT_PROMPT.format(task_description=task_description)

        # Run the coding agent
        if config.agent_type == "claude":
            # Claude Code CLI: `claude -p <prompt>` for non-interactive mode
            agent_cmd = ["claude", "-p", prompt, "--output-format", "text"]
            if config.model:
                agent_cmd.extend(["--model", config.model])
            agent_ps = sandbox.exec(
                *agent_cmd,
                secrets=secrets,
                workdir="/workspace",
            )
        else:  # codex
            # Use OpenAI Responses API via Python script (the Codex CLI
            # requires OAuth login, incompatible with headless sandboxes)
            agent_script_f = sandbox.open("/workspace/_run_agent.py", "w")
            agent_script_f.write(_OPENAI_AGENT_SCRIPT)
            agent_script_f.close()
            agent_ps = sandbox.exec(
                "python",
                "/workspace/_run_agent.py",
                config.resolved_model,
                prompt,
                secrets=secrets,
                workdir="/workspace",
            )

        agent_ps.wait()
        elapsed = time.time() - start_time
        agent_stdout = agent_ps.stdout.read()
        agent_stderr = agent_ps.stderr.read()
        agent_rc = agent_ps.returncode

        # Log agent execution details for debugging
        print(f"[agents] Agent exited with rc={agent_rc} in {elapsed:.1f}s")
        if agent_stderr.strip():
            print(f"[agents] stderr: {agent_stderr[:500]}")
        if agent_stdout.strip():
            print(f"[agents] stdout length: {len(agent_stdout)} chars")

        # Try to read the solution file the agent was asked to create
        try:
            read_ps = sandbox.exec("cat", "/workspace/solution.py")
            read_ps.wait()
            solution = read_ps.stdout.read()
            if solution.strip():
                return solution.strip(), elapsed
        except Exception:
            pass

        # Fallback: look for .py files the agent may have created
        try:
            ls_ps = sandbox.exec(
                "find",
                "/workspace",
                "-name",
                "*.py",
                "-not",
                "-path",
                "*/__pycache__/*",
                "-not",
                "-name",
                "_run_agent.py",
            )
            ls_ps.wait()
            py_files = ls_ps.stdout.read().strip().split("\n")
            for py_file in py_files:
                if py_file.strip():
                    cat_ps = sandbox.exec("cat", py_file.strip())
                    cat_ps.wait()
                    content = cat_ps.stdout.read()
                    if content.strip():
                        return content.strip(), elapsed
        except Exception:
            pass

        # Last resort: extract code from agent stdout
        if agent_stdout.strip():
            return extract_code(agent_stdout), elapsed

        raise RuntimeError("Agent produced no solution file and no code in output")

    finally:
        sandbox.terminate()
