import csv
import json
from pathlib import Path

import modal
from common import app, dataset_volume

image = (
    modal.Image.debian_slim().pip_install("pandas").add_local_python_source("common")
)


@app.function(
    volumes={"/data": dataset_volume},
    image=image,
)
def upload_token_counts():
    metadata_path = Path("/data/raw/metadata.csv")
    output_path = Path("/data/token_counts.json")

    token_counts = {}

    with metadata_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 3:
                continue
            wav_filename = f"{row[0]}.wav"
            normalized_transcription = row[2]
            token_count = len(normalized_transcription.strip().split())
            token_counts[wav_filename] = token_count

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(token_counts, f, indent=2)

    print(f"✅ Wrote token_counts.json with {len(token_counts)} entries.")
