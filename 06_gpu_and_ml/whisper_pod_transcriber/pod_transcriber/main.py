"""
whisper-pod-transcriber uses OpenAI's Whisper modal to do speech-to-text transcription
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
volume = modal.SharedVolume().persist("dataset-cache-vol")

app_image = (
    modal.Image.debian_slim()
    .pip_install(
        "https://github.com/openai/whisper/archive/9f70a352f9f8630ab3aa0d06af5cb9532bd8c21d.tar.gz",
        "dacite",
        "jiwer",
        "ffmpeg-python",
        "gql[all]~=3.0.0a5",
        "pandas",
        "loguru==0.6.0",
        "torchaudio==0.12.1",
    )
    .apt_install("ffmpeg")
    .pip_install("ffmpeg-python")
)
search_image = modal.Image.debian_slim().pip_install(
    "scikit-learn~=0.24.2",
    "tqdm~=4.46.0",
    "numpy~=1.23.3",
    "dacite",
)

stub = modal.Stub(
    "whisper-pod-transcriber",
    image=app_image,
    secrets=[modal.Secret.from_name("podchaser")],
)

stub.in_progress = modal.Dict()


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def get_episode_metadata_path(podcast_id: str, guid_hash: str) -> pathlib.Path:
    return config.PODCAST_METADATA_DIR / podcast_id / f"{guid_hash}.json"


def get_transcript_path(guid_hash: str) -> pathlib.Path:
    return config.TRANSCRIPTIONS_DIR / f"{guid_hash}.json"


@stub.function(shared_volumes={config.CACHE_DIR: volume})
def populate_podcast_metadata(podcast_id: str):
    from gql import gql

    metadata_dir = config.PODCAST_METADATA_DIR / podcast_id
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = config.PODCAST_METADATA_DIR / podcast_id / "metadata.json"
    pod_metadata: podcast.PodcastMetadata = podcast.fetch_podcast(
        gql, podcast_id
    )

    with open(metadata_path, "w") as f:
        json.dump(dataclasses.asdict(pod_metadata), f)

    episodes = fetch_episodes.call(
        show_name=pod_metadata.title, podcast_id=podcast_id
    )

    for ep in episodes:
        metadata_path = get_episode_metadata_path(podcast_id, ep.guid_hash)
        with open(metadata_path, "w") as f:
            json.dump(dataclasses.asdict(ep), f)

    logger.info(f"Populated metadata for {pod_metadata.title}")


@stub.asgi(
    mounts=[modal.Mount("/assets", local_dir=config.ASSETS_PATH)],
    shared_volumes={config.CACHE_DIR: volume},
    keep_warm=2,
)
def fastapi_app():
    import fastapi.staticfiles

    from .api import web_app

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )

    return web_app


@stub.function(
    image=app_image,
)
def search_podcast(name):
    from gql import gql

    logger.info(f"Searching for '{name}'")
    client = podcast.create_podchaser_client()
    podcasts_raw = podcast.search_podcast_name(
        gql, client, name, max_results=10
    )
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


@stub.function(
    image=search_image,
    shared_volumes={config.CACHE_DIR: volume},
    timeout=(15 * 60),
)
def index():
    import dataclasses
    from collections import defaultdict

    import dacite

    logger.info("Starting transcript indexing process.")
    config.SEARCH_DIR.mkdir(parents=True, exist_ok=True)

    episodes = defaultdict(list)
    guid_hash_to_episodes = {}

    for pod_dir in config.PODCAST_METADATA_DIR.iterdir():
        if not pod_dir.is_dir():
            continue

        for file in pod_dir.iterdir():
            if file.name == "metadata.json":
                continue

            with open(file, "r") as f:
                data = json.load(f)
                ep = dacite.from_dict(
                    data_class=podcast.EpisodeMetadata, data=data
                )
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

    logger.info(
        f"Matched {len(search_records)} transcripts to episode records."
    )

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


@stub.function(
    schedule=modal.Period(hours=4),
    timeout=(30 * 60),
)
def refresh_index():
    logger.info(f"Running scheduled index refresh at {utc_now()}")
    index.call()


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
    if duration > cur_start and (duration - cur_start) > min_segment_length:
        yield cur_start, duration
        num_segments += 1
    logger.info(f"Split {path} into {num_segments} segments")


@stub.function(
    image=app_image,
    shared_volumes={config.CACHE_DIR: volume},
    cpu=2,
)
def transcribe_segment(
    start: float,
    end: float,
    audio_filepath: pathlib.Path,
    model: config.ModelSpec,
):
    import tempfile
    import time

    import ffmpeg
    import torch
    import whisper

    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        (
            ffmpeg.input(str(audio_filepath))
            .filter("atrim", start=start, end=end)
            .output(f.name)
            .overwrite_output()
            .run(quiet=True)
        )

        use_gpu = torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"
        model = whisper.load_model(
            model.name, device=device, download_root=config.MODEL_DIR
        )
        result = model.transcribe(f.name, language="en", fp16=use_gpu)  # type: ignore

    logger.info(
        f"Transcribed segment {start:.2f} to {end:.2f} of {end - start:.2f} in {time.time() - t0:.2f} seconds."
    )

    # Add back offsets.
    for segment in result["segments"]:
        segment["start"] += start
        segment["end"] += start

    return result


@stub.function(
    image=app_image,
    shared_volumes={config.CACHE_DIR: volume},
    timeout=900,
)
def transcribe_episode(
    audio_filepath: pathlib.Path,
    result_path: pathlib.Path,
    model: config.ModelSpec,
):
    segment_gen = split_silences(str(audio_filepath))

    output_text = ""
    output_segments = []
    for result in transcribe_segment.starmap(
        segment_gen, kwargs=dict(audio_filepath=audio_filepath, model=model)
    ):
        output_text += result["text"]
        output_segments += result["segments"]

    result = {
        "text": output_text,
        "segments": output_segments,
        "language": "en",
    }

    logger.info(f"Writing openai/whisper transcription to {result_path}")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)


@stub.function(
    image=app_image,
    shared_volumes={config.CACHE_DIR: volume},
    timeout=900,
)
def process_episode(podcast_id: str, episode_id: str):
    import dacite
    import whisper

    from modal import container_app

    try:
        # pre-download the model to the cache path, because the _download fn is not
        # thread-safe.
        model = config.DEFAULT_MODEL
        whisper._download(whisper._MODELS[model.name], config.MODEL_DIR, False)

        config.RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        config.TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

        metadata_path = get_episode_metadata_path(podcast_id, episode_id)
        with open(metadata_path, "r") as f:
            data = json.load(f)
            episode = dacite.from_dict(
                data_class=podcast.EpisodeMetadata, data=data
            )

        destination_path = config.RAW_AUDIO_DIR / episode_id
        podcast.store_original_audio(
            url=episode.original_download_link,
            destination=destination_path,
        )

        logger.info(
            f"Using the {model.name} model which has {model.params} parameters."
        )
        logger.info(f"Wrote episode metadata to {metadata_path}")

        transcription_path = get_transcript_path(episode.guid_hash)
        if transcription_path.exists():
            logger.info(
                f"Transcription already exists for '{episode.title}' with ID {episode.guid_hash}."
            )
            logger.info("Skipping transcription.")
        else:
            transcribe_episode.call(
                audio_filepath=destination_path,
                result_path=transcription_path,
                model=model,
            )
    finally:
        del container_app.in_progress[episode_id]

    return episode


@stub.function(
    image=app_image,
    shared_volumes={config.CACHE_DIR: volume},
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


@stub.local_entrypoint
def search_entrypoint(name: str):
    # To search for a podcast, run:
    # modal run whisper_pod_transcriber/main.py --name "search string"
    for pod in search_podcast.call(name):
        print(pod)
