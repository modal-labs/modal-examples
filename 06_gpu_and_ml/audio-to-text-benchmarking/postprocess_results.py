# ---
# lambda-test: false
# ---

import modal
from common import DATASET_VOLUME_NAME, app, dataset_volume

image = (
    modal.Image.debian_slim()
    .pip_install("pandas==2.2.3", "matplotlib==3.10.3")
    .add_local_python_source("common")
)

pricing_per_second = {
    "A10G": 0.000306,
}


@app.function(volumes={"/data": dataset_volume}, timeout=300, image=image)
def postprocess_results():
    import json
    import re
    import time
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd

    # Set up remote paths
    result_dir = Path("/data/results")
    timestamp = int(time.time())
    output_dir = Path(f"/data/analysis/{timestamp}")
    output_dir.parents[0].mkdir(
        exist_ok=True
    )  # Create /data/analysis if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    def get_latest_files(result_dir):
        files_by_model = {}
        for f in result_dir.glob("result_*.jsonl"):
            match = re.match(r"result_(.+?)_(\d+)\.jsonl", f.name)
            if not match:
                continue
            model_name, timestamp = match.groups()
            timestamp = int(timestamp)
            if (
                model_name not in files_by_model
                or timestamp > files_by_model[model_name][1]
            ):
                files_by_model[model_name] = (f, timestamp)
        return [f for f, _ in files_by_model.values()]

    def load_data(files):
        token_map_path = Path("/data/token_counts.json")
        token_map = {}
        if token_map_path.exists():
            with token_map_path.open("r") as f:
                token_map = json.load(f)

        records = []
        for file in files:
            with file.open() as f:
                for line in f:
                    data = json.loads(line)
                    model = data["model"]
                    t_time = data["transcription_time"]
                    a_dur = data["audio_duration"]
                    transcription = data.get("transcription", "")

                    filename = data.get("filename")  # e.g., "LJ001-0001.wav"

                    # Get token count from metadata file if possible
                    token_count = token_map.get(filename)
                    if token_count is None:
                        token_count = len(
                            transcription.strip().split()
                        )  # fallback estimate

                    cost = t_time * pricing_per_second["A10G"]
                    cost_per_token = cost / token_count if token_count > 0 else None
                    cost_per_audio_second = t_time * pricing_per_second["A10G"] / a_dur

                    records.append(
                        {
                            "model": model,
                            "filename": filename,
                            "transcription_time": t_time,
                            "audio_duration": a_dur,
                            "time_per_second": t_time / a_dur if a_dur > 0 else None,
                            "token_count": token_count,
                            "estimated_cost_usd": cost,
                            "cost_per_token_usd": cost_per_token,
                            "cost_per_audio_second_usd": cost_per_audio_second,
                        }
                    )
        return pd.DataFrame(records)

    def compute_statistics(df):
        return df.groupby("model").agg(
            {
                "transcription_time": ["mean", "median", "std"],
                "audio_duration": ["mean", "median", "std"],
                "time_per_second": ["mean", "median", "std"],
                "estimated_cost_usd": ["mean", "median", "sum"],
                "cost_per_token_usd": ["mean", "median"],
                "token_count": ["mean", "sum"],
            }
        )

    def plot_metrics(df, output_dir):
        def add_median_labels(ax, column):
            medians = df.groupby("model")[column].median()
            positions = range(1, len(medians) + 1)
            for pos, (label, median) in zip(positions, medians.items()):
                ax.text(
                    pos,
                    median,
                    f"{median:.5f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="black",
                )

        def save_plot(name):
            plt.tight_layout()
            plt.savefig(output_dir / f"{name}.png")
            plt.close()

        # --- Time per second of audio ---
        plt.figure(figsize=(10, 5))
        df.boxplot(column="time_per_second", by="model")
        plt.title("Transcription time per second of audio")
        plt.ylabel("Seconds of compute / second of audio")
        plt.xticks(rotation=45)
        ax1 = df.boxplot(column="transcription_time", by="model", return_type="axes")[
            "transcription_time"
        ]
        add_median_labels(ax1, "transcription_time")
        save_plot("time_per_second_boxplot")

        # --- Raw transcription time ---
        plt.figure(figsize=(10, 5))
        df.boxplot(column="transcription_time", by="model")
        plt.title("Raw Transcription Time by Model")
        plt.ylabel("Total Transcription Time (s)")
        plt.xticks(rotation=45)
        ax1 = df.boxplot(column="transcription_time", by="model", return_type="axes")[
            "transcription_time"
        ]
        add_median_labels(ax1, "transcription_time")
        save_plot("transcription_time_boxplot")

        # --- Cost per token ---
        plt.figure(figsize=(10, 5))
        df.boxplot(column="cost_per_token_usd", by="model")
        plt.title("Estimated Cost per Token by Model")
        plt.ylabel("USD per token")
        plt.xticks(rotation=45)
        ax1 = df.boxplot(column="cost_per_token_usd", by="model", return_type="axes")[
            "cost_per_token_usd"
        ]
        add_median_labels(ax1, "cost_per_token_usd")
        save_plot("cost_per_token_boxplot")

        # --- Cost per second of audio ---
        plt.figure(figsize=(10, 5))
        df.boxplot(column="cost_per_audio_second_usd", by="model")
        plt.title("Estimated Cost per Second of Audio by Model")
        plt.ylabel("USD per second of audio")
        plt.xticks(rotation=45)
        ax1 = df.boxplot(
            column="cost_per_audio_second_usd", by="model", return_type="axes"
        )["cost_per_audio_second_usd"]
        add_median_labels(ax1, "cost_per_audio_second_usd")
        save_plot("cost_per_audio_second_boxplot")

    # Run pipeline
    latest_files = get_latest_files(result_dir)
    print(f"Processing files: {[f.name for f in latest_files]}")
    df = load_data(latest_files)
    print(f"Total entries found: {len(df)}")
    stats = compute_statistics(df)

    # Save stats to both locations
    stats.to_csv(output_dir / "transcription_stats.csv")

    # Generate plots in both locations
    plot_metrics(df, output_dir)

    print("✅ Analysis complete. Results saved to:")
    print(f"  - Modal volume {DATASET_VOLUME_NAME}: {output_dir}")

    # Also save the raw data for reference
    df.to_csv(output_dir / "raw_data.csv")

    return str(output_dir)
