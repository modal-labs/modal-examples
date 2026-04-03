# ---
# lambda-test: false  # missing-secret
# ---

# # Monitor 404s on Modal docs pages

# This example uses [Datadog's Log Analytics API](https://docs.datadoghq.com/api/latest/logs/)
# to find the most common 404 paths under `/docs/` on Modal's public site,
# then posts a weekly summary to Slack. It is deployed as a
# [scheduled Modal function](https://modal.com/docs/guide/cron) that runs
# every Monday morning.

# The analysis was originally done by hand for
# [modal-labs/modal#37560](https://github.com/modal-labs/modal/pull/37560),
# which added redirects for the highest-traffic 404 doc pages.
# This app automates that same workflow so the team can keep tabs on
# new 404s as they appear.

# ## Setup

# You will need two [Modal Secrets](https://modal.com/secrets):

# 1. **`datadog`** with keys `DD_API_KEY` and `DD_APP_KEY`
#    ([Datadog API/app key docs](https://docs.datadoghq.com/account_management/api-app-keys/)).
# 2. **`docs-404-slack`** with key `SLACK_BOT_TOKEN`
#    and, optionally, `SLACK_CHANNEL` (defaults to `#docs-alerts`).

# ## Imports and app definition

import json
import os
import re
import urllib.request
from datetime import datetime, timedelta, timezone

import modal

app = modal.App("example-docs-404-monitor")

# We only need `slack-sdk` inside the container, so we bake it into the image.

slack_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "slack-sdk~=3.33"
)

# ## Querying Datadog

# The function below hits the
# [Logs Aggregate endpoint](https://docs.datadoghq.com/api/latest/logs/#aggregate-events)
# to count 404s grouped by path over the requested time window.

DATADOG_API_URL = "https://api.datadoghq.com/api/v2/logs/analytics/aggregate"

# Legitimate doc paths contain only lowercase letters, digits, hyphens,
# underscores, dots, and forward slashes.  Anything else (percent-encoded
# payloads, `${}` injections, etc.) is scanner/attack traffic and is
# filtered out.
LEGITIMATE_PATH_RE = re.compile(r"^/docs/[a-zA-Z0-9_./-]+$")


def query_datadog_404s(
    from_time: str = "now-7d",
    to_time: str = "now",
    limit: int = 50,
) -> list[dict]:
    """Return a list of ``{"path": ..., "count": ...}`` dicts, sorted
    descending by count, for 404s under ``/docs/*``."""

    api_key = os.environ["DD_API_KEY"]
    app_key = os.environ["DD_APP_KEY"]

    body = {
        "filter": {
            "query": (
                "service:nginx-public"
                " @http.status_code:404"
                " @http.url_details.path:/docs/*"
            ),
            "from": from_time,
            "to": to_time,
        },
        "compute": [{"aggregation": "count", "type": "total"}],
        "group_by": [
            {"facet": "@http.url_details.path", "limit": limit},
        ],
    }

    headers = {
        "DD-API-KEY": api_key,
        "DD-APPLICATION-KEY": app_key,
        "Content-Type": "application/json",
    }

    req = urllib.request.Request(
        DATADOG_API_URL,
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    results = []
    for bucket in data["data"]["buckets"]:
        path = bucket["by"]["@http.url_details.path"]
        count = bucket["computes"]["c0"]
        if LEGITIMATE_PATH_RE.match(path):
            results.append({"path": path, "count": count})

    results.sort(key=lambda r: r["count"], reverse=True)
    return results


# ## Formatting the report

# The Slack message is formatted with
# [Block Kit](https://api.slack.com/block-kit) for readability, but the
# fallback ``text`` field contains a plain-text version so it also works
# in notifications.

DOCS_BASE_URL = "https://modal.com"


def format_slack_message(results: list[dict], from_dt: datetime, to_dt: datetime):
    """Build a Slack Block Kit message payload from the analysis results."""

    date_range = f"{from_dt.strftime('%b %d')} - {to_dt.strftime('%b %d, %Y')}"

    if not results:
        text = f"No legitimate 404s found for docs pages ({date_range})."
        return {
            "text": text,
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f":white_check_mark: {text}"},
                }
            ],
        }

    total_404s = sum(r["count"] for r in results)
    header = (
        f":mag: *Docs 404 Report* ({date_range})\n"
        f"Found *{total_404s:,}* legitimate 404 hits "
        f"across *{len(results)}* unique paths."
    )

    lines = []
    for r in results[:25]:
        path = r["path"]
        url = f"{DOCS_BASE_URL}{path}"
        lines.append(f"*{r['count']:,}*  <{url}|`{path}`>")
    table = "\n".join(lines)

    text = f"Docs 404 Report ({date_range}): {total_404s:,} hits across {len(results)} paths"

    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": header}},
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": table}},
    ]

    if len(results) > 25:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"_Showing top 25 of {len(results)} paths._",
                    }
                ],
            }
        )

    return {"text": text, "blocks": blocks}


# ## Posting to Slack

DEFAULT_CHANNEL = "#docs-alerts"


@app.function(
    image=slack_image,
    secrets=[
        modal.Secret.from_name("docs-404-slack", required_keys=["SLACK_BOT_TOKEN"]),
    ],
)
def post_to_slack(message_payload: dict):
    """Post a Block Kit message to Slack."""
    import slack_sdk

    token = os.environ["SLACK_BOT_TOKEN"]
    channel = os.environ.get("SLACK_CHANNEL", DEFAULT_CHANNEL)

    client = slack_sdk.WebClient(token=token)
    client.chat_postMessage(
        channel=channel,
        text=message_payload["text"],
        blocks=message_payload.get("blocks"),
    )
    print(f"Posted report to {channel}")


_RELATIVE_TIME_RE = re.compile(r"^now-(\d+)([dhms])$")
_UNIT_MAP = {"d": "days", "h": "hours", "m": "minutes", "s": "seconds"}


def _parse_relative_time(spec: str, now: datetime) -> datetime:
    """Convert a Datadog-style relative time like ``now-7d`` to a datetime."""
    m = _RELATIVE_TIME_RE.match(spec)
    if not m:
        return now - timedelta(days=7)
    value, unit = int(m.group(1)), m.group(2)
    return now - timedelta(**{_UNIT_MAP[unit]: value})


# ## Running the analysis


@app.function(
    secrets=[
        modal.Secret.from_name("datadog", required_keys=["DD_API_KEY", "DD_APP_KEY"]),
    ],
)
def analyze_docs_404s(from_time: str = "now-7d", to_time: str = "now"):
    """Query Datadog for 404s, format the report, and post to Slack."""

    print(f"Querying Datadog for 404s ({from_time} .. {to_time}) ...")
    results = query_datadog_404s(from_time=from_time, to_time=to_time)

    print(f"Found {len(results)} legitimate 404 paths")
    for r in results[:10]:
        print(f"  {r['count']:>6}  {r['path']}")

    now = datetime.now(tz=timezone.utc)
    to_dt = now
    from_dt = _parse_relative_time(from_time, now)

    message = format_slack_message(results, from_dt, to_dt)
    post_to_slack.remote(message)


# ## Weekly schedule

# The function is scheduled to run every Monday at 09:00 UTC.


@app.function(schedule=modal.Cron("0 9 * * 1"))
def weekly_report():
    analyze_docs_404s.remote()


# ## Running manually

# You can also trigger the analysis manually:
#
# ```bash
# modal run docs_404_monitor.py
# ```
#
# To look further back in time:
#
# ```bash
# modal run docs_404_monitor.py --from-time now-30d
# ```
#
# Once you are happy with the output, deploy the scheduled version:
#
# ```bash
# modal deploy docs_404_monitor.py
# ```


@app.local_entrypoint()
def main(from_time: str = "now-7d", to_time: str = "now"):
    analyze_docs_404s.remote(from_time=from_time, to_time=to_time)
