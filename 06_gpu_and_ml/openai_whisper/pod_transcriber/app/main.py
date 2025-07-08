"""
parakeet-pod-transcriber uses NVIDIA's Parakeet ASR model to do speech-to-text transcription
of podcasts.
"""

import dataclasses
import datetime
import json
import pathlib
from typing import Iterator, Tuple

import modal

from . import config, podcast, search

logger = config.get_logger(__name__)

volume = modal.Volume.from_name("dataset-cache-vol", create_if_missing=True)
model_cache = modal.Volume.from_name("parakeet-model-cache", create_if_missing=True)

app_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "dacite",
        "jiwer",
        "ffmpeg-python",
        "gql[all]~=3.0.0a5",
        "pandas",
        "loguru==0.6.0",
        "fastapi[standard]==0.115.4",
        "numpy<2",
    )
    .add_local_dir(
        "app",
        "/root/app",
        ignore=~modal.FilePatternMatcher("**/*.py"),
    )
)
parakeet_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache",  # cache directory for Hugging Face models
            "DEBIAN_FRONTEND": "noninteractive",
            "CXX": "g++",
            "CC": "g++",
        }
    )
    .apt_install("ffmpeg")
    .pip_install(
        "hf_transfer==0.1.9",
        "huggingface_hub[hf-xet]==0.31.2",
        "nemo_toolkit[asr]==2.3.0",
        "cuda-python==12.8.0",
        "fastapi==0.115.12",
        "numpy<2",
        "ffmpeg-python",
    )
    .entrypoint([])  # silence chatty logs by container on start
    .add_local_dir(
        "app",
        "/root/app",
        ignore=~modal.FilePatternMatcher("**/*.py"),
    )
)


search_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "scikit-learn~=1.3.0",
        "tqdm~=4.46.0",
        "numpy~=1.23.3",
        "dacite",
    )
    .add_local_dir(
        "app",
        "/root/app",
        ignore=~modal.FilePatternMatcher("**/*.py"),
    )
)

app = modal.App(
    "parakeet-pod-transcriber",
    image=app_image,
    secrets=[modal.Secret.from_name("podchaser")],
    include_source=False,
)

in_progress = modal.Dict.from_name(
    "pod-transcriber-in-progress", create_if_missing=True
)


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def get_episode_metadata_path(podcast_id: str, guid_hash: str) -> pathlib.Path:
    return config.PODCAST_METADATA_DIR / podcast_id / f"{guid_hash}.json"


def get_transcript_path(guid_hash: str) -> pathlib.Path:
    return config.TRANSCRIPTIONS_DIR / f"{guid_hash}.json"


@app.function(volumes={config.CACHE_DIR: volume})
def populate_podcast_metadata(podcast_id: str):
    from gql import gql

    metadata_dir = config.PODCAST_METADATA_DIR / podcast_id
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = config.PODCAST_METADATA_DIR / podcast_id / "metadata.json"
    pod_metadata: podcast.PodcastMetadata = podcast.fetch_podcast(gql, podcast_id)

    with open(metadata_path, "w") as f:
        json.dump(dataclasses.asdict(pod_metadata), f)

    episodes = fetch_episodes.remote(
        show_name=pod_metadata.title, podcast_id=podcast_id
    )

    for ep in episodes:
        metadata_path = get_episode_metadata_path(podcast_id, ep.guid_hash)
        with open(metadata_path, "w") as f:
            json.dump(dataclasses.asdict(ep), f)

    volume.commit()

    logger.info(f"Populated metadata for {pod_metadata.title}")


@app.function(
    image=app_image.add_local_dir(
        config.ASSETS_PATH,
        remote_path="/assets",
    ),
    volumes={config.CACHE_DIR: volume},
    min_containers=2,
)
@modal.asgi_app()
def fastapi_app():
    import fastapi.staticfiles

    from .api import web_app

    web_app.mount("/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True))

    web_app.state.volume = volume

    return web_app


@app.function()
def search_podcast(name):
    from gql import gql

    logger.info(f"Searching for '{name}'")
    client = podcast.create_podchaser_client()
    podcasts_raw = podcast.search_podcast_name(gql, client, name, max_results=10)
    logger.info(f"Found {len(podcasts_raw)} results for '{name}'")
    return [
        podcast.PodcastMetadata(
            id=pod["id"],
            title=pod["title"],
            description=pod["description"],
            html_description=pod["htmlDescription"],
            language=pod["language"],
            web_url=pod["webUrl"],
        )
        for pod in podcasts_raw
    ]


@app.function(
    image=search_image,
    volumes={config.CACHE_DIR: volume},
    timeout=(400 * 60),
)
def refresh_index():
    import dataclasses
    from collections import defaultdict

    import dacite

    logger.info(f"Running scheduled index refresh at {utc_now()}")
    config.SEARCH_DIR.mkdir(parents=True, exist_ok=True)

    episodes = defaultdict(list)
    guid_hash_to_episodes = {}

    for pod_dir in config.PODCAST_METADATA_DIR.iterdir():
        if not pod_dir.is_dir():
            continue

        for filepath in pod_dir.iterdir():
            if filepath.name == "metadata.json":
                continue

            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
            except json.decoder.JSONDecodeError:
                logger.warning(f"Removing corrupt JSON metadata file: {filepath}.")
                filepath.unlink()

            ep = dacite.from_dict(data_class=podcast.EpisodeMetadata, data=data)
            episodes[ep.podcast_title].append(ep)
            guid_hash_to_episodes[ep.guid_hash] = ep

    logger.info(f"Loaded {len(guid_hash_to_episodes)} podcast episodes.")

    transcripts = {}
    if config.TRANSCRIPTIONS_DIR.exists():
        for file in config.TRANSCRIPTIONS_DIR.iterdir():
            with open(file, "r") as f:
                data = json.load(f)
                guid_hash = file.stem.split("-")[0]
                transcripts[guid_hash] = data

    # Important: These have to be the same length and have same episode order.
    # i-th element of indexed_episodes is the episode indexed by the i-th element
    # of search_records
    indexed_episodes = []
    search_records = []
    for key, value in transcripts.items():
        idxd_episode = guid_hash_to_episodes.get(key)
        if idxd_episode:
            search_records.append(
                search.SearchRecord(
                    title=idxd_episode.title,
                    text=value["text"],
                )
            )
            # Prepare records for JSON serialization
            indexed_episodes.append(dataclasses.asdict(idxd_episode))

    logger.info(f"Matched {len(search_records)} transcripts to episode records.")

    filepath = config.SEARCH_DIR / "all.json"
    logger.info(f"writing {filepath}")
    with open(filepath, "w") as f:
        json.dump(indexed_episodes, f)

    logger.info(
        "calc feature vectors for all transcripts, keeping track of similar podcasts"
    )
    X, v = search.calculate_tfidf_features(search_records)
    sim_svm = search.calculate_similarity_with_svm(X)
    filepath = config.SEARCH_DIR / "sim_tfidf_svm.json"
    logger.info(f"writing {filepath}")
    with open(filepath, "w") as f:
        json.dump(sim_svm, f)

    logger.info("calculate the search index to support search")
    search_dict = search.build_search_index(search_records, v)
    filepath = config.SEARCH_DIR / "search.json"
    logger.info(f"writing {filepath}")
    with open(filepath, "w") as f:
        json.dump(search_dict, f)

    volume.commit()


def split_silences(
    path: str, min_segment_length: float = 30.0, min_silence_length: float = 1.0
) -> Iterator[Tuple[float, float]]:
    """Split audio file into contiguous chunks using the ffmpeg `silencedetect` filter.
    Yields tuples (start, end) of each chunk in seconds."""

    import re

    import ffmpeg

    silence_end_re = re.compile(
        r" silence_end: (?P<end>[0-9]+(\.?[0-9]*)) \| silence_duration: (?P<dur>[0-9]+(\.?[0-9]*))"
    )

    metadata = ffmpeg.probe(path)
    duration = float(metadata["format"]["duration"])

    reader = (
        ffmpeg.input(str(path))
        .filter("silencedetect", n="-10dB", d=min_silence_length)
        .output("pipe:", format="null")
        .run_async(pipe_stderr=True)
    )

    cur_start = 0.0
    num_segments = 0

    while True:
        line = reader.stderr.readline().decode("utf-8")
        if not line:
            break
        match = silence_end_re.search(line)
        if match:
            silence_end, silence_dur = match.group("end"), match.group("dur")
            split_at = float(silence_end) - (float(silence_dur) / 2)

            if (split_at - cur_start) < min_segment_length:
                continue

            yield cur_start, split_at
            cur_start = split_at
            num_segments += 1

    # silencedetect can place the silence end *after* the end of the full audio segment.
    # Such segments definitions are negative length and invalid.
    if duration > cur_start:
        yield cur_start, duration
        num_segments += 1
    logger.info(f"Split {path} into {num_segments} segments")


# Parakeet ASR class for handling model loading and transcription
@app.cls(
    volumes={"/cache": model_cache, config.CACHE_DIR: volume},
    image=parakeet_image,
)
class ParakeetASR:
    model_name: str = modal.parameter()

    @modal.enter()
    def load(
        self,
    ):
        import logging

        import nemo.collections.asr as nemo_asr

        # silence chatty logs from nemo
        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name
        )

    @modal.batched(max_batch_size=14, wait_ms=10)
    def transcribe_segment(
        self,
        starts: list[float],
        ends: list[float],
        audio_filepaths: list[pathlib.Path],
    ):
        import tempfile
        import time
        import wave

        import ffmpeg
        import numpy as np

        t0 = time.time()

        data = []
        for start, end in zip(starts, ends):
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                (
                    ffmpeg.input(str(audio_filepaths[0]))
                    .filter("atrim", start=start, end=end)
                    .output(
                        f.name, format="wav", acodec="pcm_s16le", ac=1, ar=16000
                    )  # 16kHz mono
                    .overwrite_output()
                    .run(quiet=True)
                )

                # convert to float32 for Parakeet
                with wave.open(f.name, "rb") as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    data.append(
                        np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                    )

        hypotheses = self.model.transcribe(data, timestamps=True)

        for start, end in zip(starts, ends):
            logger.info(
                f"Transcribed segment {start:.2f} to {end:.2f} ({end - start:.2f}s duration) in {time.time() - t0:.2f} seconds."
            )

        # Add back offsets.
        lst_segment_timestamps = []
        for start, end, hypothesis in zip(starts, ends, hypotheses):
            segment_timestamps = hypothesis.timestamp["segment"]
            for stamp in segment_timestamps:
                stamp["start"] += start
                stamp["end"] += start
            lst_segment_timestamps.append(segment_timestamps)

        # Create result in Whisper-compatible format
        results = [
            {
                "text": hypothesis.text,
                "segments": [
                    {
                        "id": i,
                        "start": stamp["start"],
                        "end": stamp["end"],
                        "text": stamp["segment"],
                    }
                    for i, stamp in enumerate(segment_timestamps)
                ],
            }
            for start, end, hypothesis, segment_timestamps in zip(
                starts, ends, hypotheses, lst_segment_timestamps
            )
        ]

        return results


@app.function(
    image=app_image,
    volumes={config.CACHE_DIR: volume},
    timeout=900,
)
def transcribe_episode(
    audio_filepath: pathlib.Path,
    result_path: pathlib.Path,
    model_name: str,
):
    segment_gen = split_silences(str(audio_filepath))

    # Create an instance of ParakeetASR
    parakeet = ParakeetASR(model_name=model_name)

    output_text = ""
    output_segments = []
    for result in parakeet.transcribe_segment.starmap(
        segment_gen,
        kwargs=dict(
            audio_filepaths=audio_filepath
        ),  # `audio_filepaths` instead of `audio_filepath` since `modal.batched` is used
    ):
        output_text += result["text"]
        output_segments += result["segments"]

    result = {
        "text": output_text,
        "segments": output_segments,
        "language": "en",
    }

    logger.info(f"Writing Parakeet ASR transcription to {result_path}")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)

    volume.commit()


@app.function(
    image=app_image,
    volumes={config.CACHE_DIR: volume},
    timeout=900,
)
def process_episode(podcast_id: str, episode_id: str):
    import dacite

    try:
        model_spec = config.DEFAULT_MODEL

        config.RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        config.TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

        metadata_path = get_episode_metadata_path(podcast_id, episode_id)
        with open(metadata_path, "r") as f:
            data = json.load(f)
            episode = dacite.from_dict(data_class=podcast.EpisodeMetadata, data=data)

        destination_path = config.RAW_AUDIO_DIR / episode_id
        podcast.store_original_audio(
            url=episode.original_download_link,
            destination=destination_path,
        )

        volume.commit()

        logger.info(
            f"Using the {model_spec.name} model which has {model_spec.params} parameters."
        )
        logger.info(f"Wrote episode metadata to {metadata_path}")

        transcription_path = get_transcript_path(episode.guid_hash)
        if transcription_path.exists():
            logger.info(
                f"Transcription already exists for '{episode.title}' with ID {episode.guid_hash}."
            )
            logger.info("Skipping transcription.")
        else:
            transcribe_episode.remote(
                audio_filepath=destination_path,
                result_path=transcription_path,
                model_name=model_spec.name,
            )
    finally:
        del in_progress[episode_id]

    return episode


@app.function(
    image=app_image,
    volumes={config.CACHE_DIR: volume},
)
def fetch_episodes(show_name: str, podcast_id: str, max_episodes=100):
    import hashlib

    from gql import gql

    client = podcast.create_podchaser_client()
    episodes_raw = podcast.fetch_episodes_data(
        gql, client, podcast_id, max_episodes=max_episodes
    )
    logger.info(f"Retrieved {len(episodes_raw)} raw episodes")
    episodes = [
        podcast.EpisodeMetadata(
            podcast_id=podcast_id,
            podcast_title=show_name,
            title=ep["title"],
            publish_date=ep["airDate"],
            description=ep["description"],
            episode_url=ep["url"],
            html_description=ep["htmlDescription"],
            guid=ep["guid"],
            guid_hash=hashlib.md5(ep["guid"].encode("utf-8")).hexdigest(),
            original_download_link=ep["audioUrl"],
        )
        for ep in episodes_raw
        if "guid" in ep and ep["guid"] is not None
    ]
    no_guid_count = len(episodes) - len(episodes_raw)
    logger.info(f"{no_guid_count} episodes had no GUID and couldn't be used.")
    return episodes


@app.local_entrypoint()
def search_entrypoint(name: str):
    # To search for a podcast, run:
    # modal run -m app.main --name "search string"
    for pod in search_podcast.remote(name):
        print(pod)
