# ---
# cmd: ["modal", "run", "13_sandboxes/browserbase_cli.py"]
# lambda-test: false  # missing-secret
# pytest: false
# ---

# # Run a browser agent in a Modal Sandbox

# A Modal [Sandbox](https://modal.com/docs/guide/sandbox) is a great place to run an
# **agent's tools**. This example keeps the browser itself somewhere else, on
# [Browserbase](https://www.browserbase.com) — a browser Browserbase runs and
# maintains, with a real fingerprint recognized as legitimate by bot protection
# partners and, on [Verified](https://www.browserbase.com/verified) sessions,
# automatic CAPTCHA solving. Browserbase handles all of that, so wiring a browser into
# your existing agent harness as a tool is a CLI call away instead of a separate piece
# of infrastructure to build and operate.

# The Sandbox holds a single tool, the
# [`browse`](https://github.com/browserbase/stagehand/tree/main/packages/cli) CLI, and
# an Anthropic agent loop drives it via Claude's native **bash tool**. Each bash command
# runs inside the Sandbox with `sandbox.exec`, and `browse` connects out over CDP to the
# remote browser running on Browserbase.

# ```
# ┌─────────────────────┐   bash tool    ┌──────────────────┐  CDP over wss  ┌────────────────────────┐
# │  Modal Function     │ ─────────────▶ │  Modal Sandbox   │ ─────────────▶ │  Browserbase browser   │
# │  Claude agent loop  │  sandbox.exec  │  `browse` CLI    │ ◀───────────── │  (the real Chrome)     │
# └─────────────────────┘ ◀───────────── └──────────────────┘   page data    └────────────────────────┘
# ```
# The agent's task (overridable): research the current top mechanical keyboards on
# Amazon and return a comparison of each product's title, price, star rating, and
# number of ratings. Amazon search renders its product grid client-side and returns
# nothing useful to a plain `curl`, so the agent has to drive a real browser — which
# is exactly what `browse` gives it.

# To run it, first create the `browser-agent` Secret described below, then:

# ```bash
# modal run 13_sandboxes/browserbase_cli.py
# ```


from datetime import datetime, timezone

import modal

# ## Build the Sandbox image

# We build the Sandbox image ourselves from the official
# [`node:20-slim`](https://hub.docker.com/_/node) base and install the
# [`browse`](https://github.com/browserbase/stagehand/tree/main/packages/cli)
# CLI with `npm`, pinned to an explicit version. The agent loop itself runs in
# a Modal Function and only needs `anthropic`, so we add that to the
# Function's image separately below.

BROWSE_VERSION = "0.9.5"  # pin to avoid surprises!

sandbox_image = modal.Image.from_registry(
    "node:20-slim", add_python="3.12"
).run_commands(f"npm install -g browse@{BROWSE_VERSION} && npm cache clean --force")

app = modal.App("example-browserbase-cli")

MINUTES = 60  # seconds, for readable timeouts

# ## Manage credentials

# Both the agent loop and the Sandbox need an Anthropic key and a Browserbase key.
# We store them once as a named Modal Secret and reference it by name. Create it
# before your first run with:

# ```bash
# modal secret create browser-agent ANTHROPIC_API_KEY=... BROWSERBASE_API_KEY=...
# ```

# Or add it via the dashboard: https://modal.com/secrets.

agent_secret = modal.Secret.from_name(
    "browser-agent", required_keys=["ANTHROPIC_API_KEY", "BROWSERBASE_API_KEY"]
)

# ## The agent

# The model's only tool is Claude's built-in **bash tool**
# (`{"type": "bash_20250124", "name": "bash"}`) — a schema-less tool whose commands we
# execute ourselves. We run each command inside the Sandbox with `sandbox.exec`, so the
# model gets a real shell whose one useful program is `browse`.

# We don't hand-roll a custom `browse` tool or inject flags into the model's commands.
# Instead the Sandbox's environment does the steering: `BROWSERBASE_API_KEY` makes
# `browse` default to a **remote** Browserbase browser, and `BROWSE_SESSION=agent`
# scopes every call to one shared session, so the agent keeps the same browser tab
# across commands.

# The system prompt is deliberately minimal: it doesn't enumerate any `browse`
# subcommands. It just tells the model the CLI is installed and pre-configured and
# points it at `browse skills show` (the CLI's bundled usage guide), with
# `browse --help` as a supplementary reference, to learn the interface itself, then
# lets it plan its own steps for whatever task it's given. It also reminds the model
# it's running in a Sandbox with no local browser (so it never passes `--local`),
# injects today's date at call time so it never guesses or hardcodes one, and tells
# it to give its best final answer from whatever it's gathered if it's nearing the
# step limit.

BROWSE_SESSION = "agent"
DEFAULT_TASK = (
    "Using Amazon (https://www.amazon.com), research the current top mechanical keyboards: "
    "search the site, then for the top 5 results compare each product's title, price, star "
    "rating, and number of ratings. Return a comparison table including each product's URL."
)
MODEL = "claude-sonnet-5"
MAX_STEPS = 40
MAX_OUTPUT_CHARS = 40_000  # cap each tool result so it fits the context budget


# Built fresh on every invocation (not a module-level constant) so the injected
# date is always today, even if Modal reuses a warm container across runs.
def build_system_prompt() -> str:
    today = datetime.now(timezone.utc).date()
    return (
        "You are an autonomous deep-research agent. You have a `browse` CLI (Browserbase browser automation) "
        "in your bash tool — it is installed, and its auth and a shared browser session are already configured "
        "via environment variables. Learn how to use it by running `browse skills show` (it prints the CLI's "
        "bundled usage guide); use `browse --help` (and `browse <command> --help`) for extra detail on any "
        "command. Then complete the task.\n\n"
        "You are running inside a Modal Sandbox with no local browser — every `browse` session is remote and "
        f"already configured via environment variables, so never pass `--local`. Today's date is {today}; use "
        "it for anything time-sensitive and never hardcode or guess a date. If you're nearing the step limit, "
        "stop navigating and give the best final answer you can from what you've already gathered.\n\n"
        "When you cite a document, link the direct document itself, not a viewer, preview, or index page that "
        "wraps it. Return a clear, well-sourced answer."
    )


# Claude's native bash tool. The schema is built into the model — we declare it by
# type and name only, and our handler executes the `command` Claude sends.

BASH_TOOL = {"type": "bash_20250124", "name": "bash"}

agent_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "anthropic==0.115.0"
)


@app.function(
    image=agent_image,
    secrets=[agent_secret],
)
def run_agent(task: str = DEFAULT_TASK) -> str:
    import anthropic

    # Boot the Sandbox. Its image's default CMD exits immediately, so we pass a
    # long-lived no-op entrypoint and drive it entirely with `sandbox.exec`.
    sandbox = modal.Sandbox.create(
        "sleep",
        "infinity",
        app=app,
        image=sandbox_image,
        secrets=[agent_secret],
        env={"BROWSE_SESSION": BROWSE_SESSION},
        timeout=15 * MINUTES,
    )
    print(f"Sandbox ID: {sandbox.object_id}")

    def run_bash(command: str) -> str:
        """Execute one bash command inside the Sandbox and return its output."""
        print(f"-> {command}")
        proc = sandbox.exec("bash", "-c", command, timeout=60)
        proc.wait()
        out = (proc.stdout.read() or "") + (proc.stderr.read() or "")
        print(f"   <- {len(out)} chars")
        return out[:MAX_OUTPUT_CHARS] or "(no output)"

    client = anthropic.Anthropic()

    # The agent loop: call the model, run any bash commands it requests, feed the
    # results back as tool_result blocks, and repeat until it stops calling tools
    # (or we hit the step cap).
    messages = [{"role": "user", "content": task}]
    final_text = ""

    try:
        for _ in range(MAX_STEPS):
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=build_system_prompt(),
                tools=[BASH_TOOL],
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            text = "".join(b.text for b in response.content if b.type == "text")
            if text:
                final_text = text

            if response.stop_reason != "tool_use" or not tool_uses:
                break

            # Run every requested command, returning all results in ONE user message.
            tool_results = []
            for block in tool_uses:
                if block.input.get("restart"):
                    output = (
                        "(bash session restart is a no-op; each command runs fresh)"
                    )
                else:
                    output = run_bash(block.input["command"])
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )
            messages.append({"role": "user", "content": tool_results})
    finally:
        # Close the remote browser session, then terminate the Sandbox.
        run_bash("browse stop")
        sandbox.terminate()

    answer = final_text or "(no answer)"
    print("\n===== FINAL ANSWER =====\n" + answer)
    return answer


# ## Run it

# `run_agent` is the only Function in this file, so `modal run
# 13_sandboxes/browserbase_cli.py` invokes it directly — no separate
# `local_entrypoint` needed. Pass a different goal with `--task "..."`.

# > **Note on protected sites.** This example uses a plain remote browser, which works
# > on any Browserbase plan. To reach sites behind aggressive bot-detection,
# > Browserbase also offers Verified browsers (residential IP + automatic CAPTCHA
# > solving), which require [a Scale plan](https://www.browserbase.com/pricing).
