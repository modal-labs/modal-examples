import os
from datetime import datetime
from pathlib import Path

import modal

if modal.is_local():
    workspace = modal.config._profile
else:
    workspace = os.environ["MODAL_WORKSPACE"]


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("locust~=2.29.1", "openai~=1.37.1")
    .env({"MODAL_WORKSPACE": workspace})
    .add_local_file(
        Path(__file__).parent / "locustfile.py",
        remote_path="/root/locustfile.py",
    )
)
volume = modal.Volume.from_name(
    "loadtest-vllm-oai-results", create_if_missing=True
)
remote_path = Path("/root") / "loadtests"
OUT_DIRECTORY = (
    remote_path / datetime.utcnow().replace(microsecond=0).isoformat()
)

app = modal.App("loadtest-vllm-oai", image=image, volumes={remote_path: volume})

workers = 8
host = f"https://{workspace}--example-vllm-openai-compatible-serve.modal.run"
csv_file = OUT_DIRECTORY / "stats.csv"
default_args = [
    "-H",
    host,
    "--processes",
    str(workers),
    "--csv",
    csv_file,
]

MINUTES = 60  # seconds


@app.function(allow_concurrent_inputs=1000, cpu=workers)
@modal.web_server(port=8089)
def serve():
    run_locust.local(default_args)


@app.function(cpu=workers, timeout=60 * MINUTES)
def run_locust(args: list, wait=False):
    import subprocess

    process = subprocess.Popen(["locust"] + args)
    if wait:
        process.wait()
        return process.returncode


@app.local_entrypoint()
def main(
    r: float = 1.0,
    u: int = 36,
    t: str = "1m",  # no more than the timeout of run_locust, one hour
):
    args = default_args + [
        "--spawn-rate",
        str(r),
        "--users",
        str(u),
        "--run-time",
        t,
    ]

    html_report_file = OUT_DIRECTORY / "report.html"
    args += [
        "--headless",  # run without browser UI
        "--autostart",  # start test immediately
        "--autoquit",  # stop once finished...
        "10",  # ...but wait ten seconds
        "--html",  # output an HTML-formatted report
        html_report_file,  # to this location
    ]

    if exit_code := run_locust.remote(args, wait=True):
        SystemExit(exit_code)
    else:
        print("finished successfully")
