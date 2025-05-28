# ---
# lambda-test: false
# ---

import time
from pathlib import Path
import json
from common import COLOR


def write_results(results: list[tuple[str, str, float, float]], model_name: str):
    timestamp = int(time.time())
    result_path = Path(f"result_{model_name}_{timestamp}.jsonl")
    with result_path.open("w") as f:
        for filename, transcription, transcription_time, duration in results:
            row = {
                "model": model_name,
                "filename": filename,
                "transcription": transcription,
                "transcription_time": transcription_time,
                "audio_duration": duration,
            }
            f.write(json.dumps(row) + "\n")
    return result_path


def print_header(text):
    print(f"{COLOR['HEADER']}{text}{COLOR['ENDC']}")


def print_error(text):
    print(f"{COLOR['ERROR']}{text}{COLOR['ENDC']}")
