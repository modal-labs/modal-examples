# ---
# cmd: ["modal", "run", "--detach", "06_gpu_and_ml/llm-serving/vllm_throughput.py"]
# ---

# # Run LLM inference at maximum throughput

# This example demonstrates some techniques for running LLM inference
# at the highest possible throughput on Modal.

# For more on other aspects of maximizing the performance of LLM inference, see
# [our guide](https://modal.com/docs/guide/high-performance-llm-inference).

# As our sample application, we use an LLM to summarize thousands of filings with
# the U.S. federal government's Securities and Exchange Commission (SEC),
# made available to the public for free in daily data dumps
# via the SEC's Electronic Data Gathering, Analysis, and Retrieval System
# ([EDGAR](https://www.sec.gov/submit-filings/about-edgar)).
# We like to check out the [Form 4s](https://www.sec.gov/files/form4data.pdf),
# which detail (legal) insider trading.

# Using the Qwen 3 8B parameter LLM on this task,
# which has inputs that average a few thousand tokens
# and outputs that average a few hundred tokens,
# we observe processing speeds of ~30,000 input tok/s
# and ~2,000 output tok/s per H100 GPU.

# At Modal's [current rates](https://modal.com/pricing) as of early 2026,
# that's roughly 4¢ per million tokens.
# [According to Artificial Analysis](https://artificialanalysis.ai/models/qwen3-8b-instruct),
# API providers charge roughly five times as much for the same workload.

# ## Organizing a batch job on Modal

import datetime as dt
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

app = modal.App("example-vllm-throughput")


@app.local_entrypoint()
def main(lookback: int = 7, wait_for_results: bool = True):
    jobs = orchestrate(lookback=lookback)  # trigger remote jobs, remotely

    if wait_for_results:
        print("Collecting results locally")
        batches = modal.FunctionCall.gather(*jobs)
        for batch in batches:
            print(*(result.summary for result in batch if result.form == "4"), sep="\n")
            print("\n")
        print("Done")
    else:
        print("Collect results asynchronously with modal.FunctionCall.from_id")
        print("FunctionCall IDs:", *[job.object_id for job in jobs], sep="\n\t")


MINUTES = 60  # seconds
HOURS = 60 * MINUTES


def orchestrate(lookback: int) -> list[modal.FunctionCall]:
    llm = Vllm()

    today = datetime.now(tz=ZoneInfo("America/New_York")).date()  # Eastern Time
    print(f"Loading SEC filing data for the last {lookback} days")
    folders = list(extract.map(today - dt.timedelta(days=ii) for ii in range(lookback)))
    folders = filter(  # drop days with no data (weekends, holidays)
        lambda f: f is not None, folders
    )

    print("Transforming raw SEC filings")
    filing_batches = list(transform.map(folders))
    n_filings = sum(map(len, filing_batches))

    print(f"Submitting {n_filings} SEC filings to LLM for summarization")
    jobs = list(llm.process.spawn(batch) for batch in filing_batches)

    return jobs


# ## Configuring vLLM for maximum throughput

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.13")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster model transfers
)


@app.cls(
    image=vllm_image,
    gpu="h100",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
class Vllm:
    @modal.enter()
    def start(self):
        import vllm

        self.llm = vllm.LLM(
            model="Qwen/Qwen3-8B-FP8",
            max_model_len=4096 * 4,
            attention_backend="flashinfer",
            async_scheduling=True,
        )
        self.sampling_params = self.llm.get_default_sampling_params()
        self.sampling_params.max_tokens = 1000

        self.llm.chat([{"role": "user", "content": "Is this thing on?"}])

    @modal.method()
    def process(self, filings: list | None = None):
        if filings is None:
            return

        messages = [
            [
                {
                    "role": "user",
                    "content": f"/no_think Summarize this SEC filing in a single, short paragraph.\n\n{filing.text}",
                }
            ]
            for filing in filings
        ]

        start = time.time()
        responses = self.llm.chat(messages, sampling_params=self.sampling_params)
        duration_s = time.time() - start

        in_token_count = sum(len(response.prompt_token_ids) for response in responses)
        out_token_count = sum(
            len(response.outputs[0].token_ids) for response in responses
        )

        print(
            f"processed {in_token_count} prompt tokens generated {out_token_count} in {int(duration_s)} seconds"
        )

        for response, filing in zip(responses, filings):
            filing.summary = response.outputs[0].text

        return filings

    @modal.exit()
    def stop(self):
        del self.llm


# ## Transforming SEC filings for batch processing


@dataclass
class Filing:
    accession_number: str | None = None
    form: str | None = None
    cik: str | None = None
    text: str | None = None
    summary: str | None = None


data_proc_image = modal.Image.debian_slim(python_version="3.13").uv_pip_install(
    "edgartools==5.8.3"
)

sec_edgar_feed = modal.Volume.from_name(
    "example-sec-edgar-feed", create_if_missing=True
)
data_root = Path("/data")


@app.function(volumes={data_root: sec_edgar_feed}, scaledown_window=5)
def transform(folder: str | None) -> list[Filing]:
    if folder is None:
        return []

    folder_path = data_root / folder
    paths = [p for p in folder_path.iterdir() if p.is_file() and p.suffix == ".nc"]

    print(f"Processing {len(paths)} filings")

    chunks: list[list[Path]] = [paths[i : i + 100] for i in range(0, len(paths), 100)]

    batches = list(_transform_filing_batch.map(chunks))

    filings = [f for batch in batches for f in batch if f is not None]

    print(f"Found documents for {len(filings)} filings out of {len(paths)}")

    return filings


@app.function(
    volumes={data_root: sec_edgar_feed}, scaledown_window=5, image=data_proc_image
)
def _transform_filing_batch(raw_filing_paths: list[Path]) -> list[Filing | None]:
    from edgar.sgml import FilingSGML

    out = []
    for raw_filing_path in raw_filing_paths:
        sgml = FilingSGML.from_source(raw_filing_path)
        text = extract_text(sgml)
        if text is None:
            out.append(None)
            continue
        out.append(
            Filing(
                accession_number=sgml.accession_number,
                form=sgml.form,
                cik=sgml.cik,
                text=text,
            )
        )
    return out


# ## Loading filings from the SEC EDGAR Feed

scraper_image = modal.Image.debian_slim(python_version="3.13").uv_pip_install(
    "requests==2.32.5"
)


@app.function(
    max_containers=10,
    volumes={data_root: sec_edgar_feed},
    retries=5,
    image=scraper_image,
    scaledown_window=5,
)
def extract(day: dt.date) -> str | None:
    target_folder = str(day)
    day_dir = data_root / target_folder

    # If the folder doesn't exist yet, try downloading the day's tarball
    if not day_dir.exists():
        print(f"Looking for data for {day} in SEC EDGAR Feed")
        ok = _download_from_sec_edgar(day, day_dir)
        if not ok:
            return None

    daily_name = f"{day:%Y%m%d}.nc.tar.gz"
    tar_path = day_dir / daily_name

    if not any(p.suffix == ".nc" for p in day_dir.iterdir()):
        print(f"Loading data for {day} from {tar_path}")
        _extract_tarfile(tar_path, day_dir)

    print(f"Data for {day} loaded")

    return target_folder


# ## Addenda

# The remainder of this code consists of utility functions and boiler plate used in the
# main code above.

# ### Utilities for transforming SEC Filings

# The code in this section is used to transform, normalize, and otherwise munge
# the raw filings downloaded from the SEC.

# For LLM serving, the most important piece here is the function to truncate
# documents. A maximum document length can be used to set a loose bound
# on the sequence length in the LLM engine configuration.


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = clean_xml(text)
    return text


def clean_xml(xml: str) -> str:
    import re

    _XMLNS_ATTR_RE = re.compile(r'\s+xmlns(:\w+)?="[^"]*"', re.I)
    _XML_DECL_RE = re.compile(r"^\s*<\?xml[^>]*\?>\s*", re.I)
    _EMPTY_TAG_RE = re.compile(r"<(\w+)([^>]*)>\s*</\1>", re.S)
    _BETWEEN_TAG_WS_RE = re.compile(r">\s+<")

    xml = xml.replace("\r\n", "\n").replace("\r", "\n").strip()

    # drop xml declaration, remove xmlns attributes
    xml = _XML_DECL_RE.sub("", xml)
    xml = _XMLNS_ATTR_RE.sub("", xml)

    # replace whitespace between tags with a single newline
    xml = _BETWEEN_TAG_WS_RE.sub("><", xml).replace("><", ">\n<")

    return xml.strip()


def truncate_head_tail(text: str, head: int = 30000, tail: int = 3000) -> str:
    if len(text) <= head + tail:
        return text
    return text[:head].rstrip() + "\n\n[...TRUNCATED...]\n\n" + text[-tail:].lstrip()


def extract_text(sgml) -> str | None:
    doc = sgml.xml()
    return truncate_head_tail(normalize_text(doc)) if doc else None


# ### Utilities for loading filings from the SEC EDGAR Feed

# The code in this section is used to load raw data from the Feed
# section of SEC EDGAR.

# Daily dumps are stored in [tar](https://www.math.utah.edu/docs/info/tar_4.html)
# archives, which the code below extracts.
# Archives for particular days are located by searching the SEC EDGAR Feed indices
# for the appropriate URL.

# For full compliance with SEC EDGAR etiquette,
# we recommend updating the `SEC_USER_AGENT` environment variable
# below with your name and email.


def _download_from_sec_edgar(day: dt.date, day_dir: Path) -> bool:
    import os

    import requests

    SEC_UA = os.environ.get("SEC_USER_AGENT", "YourName your.email@example.com")
    session = requests.Session()
    session.headers.update({"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"})

    base = "https://www.sec.gov/Archives/edgar/Feed"

    def quarter(d: dt.date) -> str:
        return f"QTR{(d.month - 1) // 3 + 1}"

    qtr = quarter(day)
    daily_name = f"{day:%Y%m%d}.nc.tar.gz"
    qtr_index = f"{base}/{day.year}/{qtr}/index.json"

    if not check_index(session, qtr_index, daily_name):
        print(f"no data for {day} in SEC EDGAR Feed")
        return False

    day_dir.mkdir(parents=True, exist_ok=True)

    tar_path = day_dir / daily_name
    if not tar_path.exists() or tar_path.stat().st_size == 0:
        url = f"{base}/{day.year}/{qtr}/{daily_name}"
        print(f"Downloading from {url}")
        print("This can take several minutes")
        _download_tar(session, url, tar_path)

    return True


def _extract_tarfile(from_tar_path, to_dir):
    import tarfile

    with tarfile.open(from_tar_path, "r:gz") as tf:
        for member in tf:
            if not (member.isfile() and member.name.endswith(".nc")):
                continue
            dest = to_dir / Path(member.name).name
            if dest.exists() and dest.stat().st_size > 0:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            dest.write_bytes(f.read())


def check_index(session, index_url, name) -> bool:
    r = session.get(index_url, timeout=30)
    if r.status_code == 404:
        return False
    r.raise_for_status()
    for it in r.json().get("directory", {}).get("item", []):
        if it.get("type") == "file" and it.get("name") == name:
            return True
    return False


def _download_tar(session, url, tar_path):
    resp = session.get(url, timeout=500)
    resp.raise_for_status()
    tmp = tar_path.with_suffix(tar_path.suffix + ".part")
    tmp.write_bytes(resp.content)
    tmp.replace(tar_path)
