# ---
# cmd: ["modal", "run", "13_sandboxes/browsecli_in_modal.py"]
# ---

# # Reach any website from a Modal Function with a Verified Browserbase browser
#
# Modal is great at running your **agent loop** — but a Sandbox can't browse the
# real web reliably. It has a **datacenter IP** (instantly blocked by
# Cloudflare/Akamai/DataDome), no anti-bot fingerprint hardening, and no way to
# solve a CAPTCHA. The usual fix — bundling Playwright + Chromium into the Image —
# still browses *from the datacenter IP*, so the hard sites stay blocked.
#
# This example keeps the browser **out** of the Function. The Modal Function runs
# the [`browse`](https://github.com/browserbase/stagehand/tree/main/packages/cli)
# CLI, which connects out over CDP to a **Verified [Browserbase](https://www.browserbase.com)
# browser** that:
#
# - uses a **residential / verified IP** — no datacenter-IP blocking
# - runs in **Verified browser mode** — passes bot-detection fingerprinting
# - **auto-solves CAPTCHAs / challenges** server-side
#
# ```
# ┌─────────────────────────┐      CDP over wss        ┌──────────────────────────┐
# │  Modal Function          │  ──────────────────────▶ │  Browserbase Verified     │
# │  node + `browse` CLI     │                           │  browser (residential IP, │
# │  your agent loop         │ ◀─────────────────────────│  stealth, CAPTCHA solve)   │
# └─────────────────────────┘      page data / refs     └──────────────────────────┘
# ```
#
# To run it:
#
# ```bash
# export BROWSERBASE_API_KEY=bb_live_...
# modal run 13_sandboxes/browsecli_in_modal.py
# ```
#
# (Or store it once as a named Modal [Secret](https://modal.com/docs/guide/secrets) —
# `modal secret create browserbase BROWSERBASE_API_KEY=...` — and swap the
# `modal.Secret.from_dict(...)` below for `modal.Secret.from_name("browserbase")`.)
#
# Note: Verified browsers require a Browserbase **Scale** plan
# (https://www.browserbase.com/pricing).

import os
import subprocess
from pathlib import Path

import modal

# ## Build the Image
#
# Modal's `debian_slim` has no Node, so we start from the official `node:20-slim`
# image, add a Python interpreter (so Modal can run its agent inside), install the
# `browse` CLI globally, and copy in the demo script. **No Chrome/Chromium is
# installed** — the browser lives on Browserbase and is reached over CDP at run time.

here = Path(__file__).parent

image = (
    modal.Image.from_registry("node:20-slim", add_python="3.12")
    .run_commands("npm install -g browse@latest", "browse --version")
    .add_local_file(here / "browsecli-demo.sh", "/app/browsecli-demo.sh", copy=True)
)

app = modal.App("example-browsecli-in-modal", image=image)

# ## The Function
#
# The Function shells out to the same `browse` commands as `browsecli-demo.sh`:
# create a Verified session (`--proxies --verified --solve-captchas`), open a
# Cloudflare-protected page over CDP, and assert we reached real content instead
# of a challenge wall.
#
# We inject Browserbase creds with `modal.Secret.from_dict`, reading them from the
# **local** environment at launch — so no pre-created Modal Secret is required.
#
# **CI guard.** Modal runs every gallery example live on each push, where no
# Browserbase key exists. In that case `from_dict` injects an empty string, and
# the guard below prints a clear "skipping" message and returns cleanly (exit 0)
# instead of failing CI. With a key present, the live run is cheap — one short
# Verified session.

browserbase_secret = modal.Secret.from_dict(
    {
        "BROWSERBASE_API_KEY": os.environ.get("BROWSERBASE_API_KEY", ""),
    }
)


@app.function(secrets=[browserbase_secret])
def reach_protected_site(target_url: str = "https://nowsecure.nl") -> int:
    if not os.environ.get("BROWSERBASE_API_KEY"):
        print(
            "[browsecli-in-modal] skipping live run (no BROWSERBASE_API_KEY). "
            "Set it in your env before `modal run`, e.g. export BROWSERBASE_API_KEY=..."
        )
        return 0

    # Run the same demo every sandbox template runs. We invoke the committed shell
    # script so the behavior is identical across providers.
    result = subprocess.run(
        ["bash", "/app/browsecli-demo.sh"],
        env={**os.environ, "TARGET_URL": target_url},
    )
    return result.returncode


# ## Local entrypoint
#
# `modal run 13_sandboxes/browsecli_in_modal.py` triggers this, which runs the
# Function in the cloud. Pass a different site with `--target-url`.


@app.local_entrypoint()
def main(target_url: str = "https://nowsecure.nl"):
    code = reach_protected_site.remote(target_url)
    if code == 0:
        print("[browsecli-in-modal] done")
    else:
        raise SystemExit(code)
