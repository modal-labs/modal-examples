"""
whisper-pod-transcriber uses OpenAI's Whisper modal to do speech-to-text transcription
of podcasts.
"""
import dataclasses
import datetime
import json
import pathlib
import sys
from typing import Iterator, List, Tuple

import modal
from fastapi import FastAPI, Request

from . import config, podcast, search

volume = modal.SharedVolume().persist("dataset-cache-vol")
app_image = (
    modal.Image.debian_slim()
    .pip_install(
        [
            "https://github.com/openai/whisper/archive/5d8d3e75a4826fe5f01205d81c3017a805fc2bf9.tar.gz",
            "dacite",
            "jiwer",
            "ffmpeg-python",
            "gql[all]~=3.0.0a5",
            "pandas",
            "loguru==0.6.0",
            "torchaudio==0.12.1",
        ]
    )
    .apt_install(
        [
            "ffmpeg",
        ]
    )
    .pip_install(["ffmpeg-python"])
)
search_image = modal.Image.debian_slim().pip_install(
    ["scikit-learn~=0.24.2", "tqdm~=4.46.0", "numpy~=1.23.3", "dacite"]
)

stub = modal.Stub(
    "whisper-pod-transcriber",
    image=app_image,
    secrets=[modal.Secret.from_name("podchaser")],
)
stub.in_progress = modal.Dict()
web_app = FastAPI()


def utc_now() -> datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def get_episode_metadata_path(podcast_id: str, guid_hash: str) -> pathlib.Path:
    return config.PODCAST_METADATA_DIR / podcast_id / f"{guid_hash}.json"


def get_transcript_path(guid_hash: str) -> pathlib.Path:
    return config.TRANSCRIPTIONS_DIR / f"{guid_hash}.json"


@web_app.get("/api/episode/{podcast_id}/{episode_guid_hash}")
async def get_episode(podcast_id: str, episode_guid_hash: str):
    episode_metadata_path = get_episode_metadata_path(podcast_id, episode_guid_hash)
    transcription_path = get_transcript_path(episode_guid_hash)

    with open(episode_metadata_path, "r") as f:
        metadata = json.load(f)

    if not transcription_path.exists():
        return dict(metadata=metadata)

    with open(transcription_path, "r") as f:
        data = json.load(f)

    return dict(metadata=metadata, segments=data["segments"])


@stub.function(shared_volumes={config.CACHE_DIR: volume})
def populate_podcast_metadata(podcast_id: str):
    from gql import gql

    metadata_dir = config.PODCAST_METADATA_DIR / podcast_id
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = config.PODCAST_METADATA_DIR / podcast_id / "metadata.json"
    pod_metadata: podcast.PodcastMetadata = podcast.fetch_podcast(gql, podcast_id)

    with open(metadata_path, "w") as f:
        json.dump(dataclasses.asdict(pod_metadata), f)

    episodes = fetch_episodes(show_name=pod_metadata.title, podcast_id=podcast_id)

    for ep in episodes:
        metadata_path = get_episode_metadata_path(podcast_id, ep.guid_hash)
        with open(metadata_path, "w") as f:
            json.dump(dataclasses.asdict(ep), f)

    print(f"Populated metadata for {pod_metadata.title}")


@web_app.get("/api/podcast/{podcast_id}")
async def get_podcast(podcast_id: str):
    pod_metadata_path = config.PODCAST_METADATA_DIR / podcast_id / "metadata.json"

    if not pod_metadata_path.exists():
        populate_podcast_metadata(podcast_id)
    else:
        # Refresh async.
        populate_podcast_metadata.submit(podcast_id)

    with open(pod_metadata_path, "r") as f:
        pod_metadata = json.load(f)

    episodes = []
    for file in (config.PODCAST_METADATA_DIR / podcast_id).iterdir():
        if file == pod_metadata_path:
            continue

        with open(file, "r") as f:
            ep = json.load(f)
            ep["transcribed"] = get_transcript_path(ep["guid_hash"]).exists()
            episodes.append(ep)

    episodes.sort(key=lambda ep: ep.get("publish_date"), reverse=True)

    return dict(pod_metadata=pod_metadata, episodes=episodes)


def is_podcast_recently_transcribed(podcast_id: str):
    if not config.COMPLETED_DIR.exists():
        return False
    completion_marker_path = config.COMPLETED_DIR / f"{podcast_id}.txt"
    return completion_marker_path.exists()


@web_app.post("/api/podcasts")
async def podcasts_endpoint(request: Request):
    import dataclasses

    form = await request.form()
    name = form["podcast"]
    podcasts_response = []
    for pod in search_podcast(name):
        data = dataclasses.asdict(pod)
        if is_podcast_recently_transcribed(pod.id):
            data["recently_transcribed"] = "true"
        else:
            data["recently_transcribed"] = "false"
        podcasts_response.append(data)
    return podcasts_response


@stub.asgi(
    mounts=[modal.Mount("/assets", local_dir=config.ASSETS_PATH)],
    shared_volumes={config.CACHE_DIR: volume},
    keep_warm=True,
)
def fastapi_app():
    import fastapi.staticfiles

    web_app.mount("/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True))

    return web_app


@stub.function(
    image=app_image,
)
def search_podcast(name):
    from gql import gql

    print(f"Searching for '{name}'")
    client = podcast.create_podchaser_client()
    podcasts_raw = podcast.search_podcast_name(gql, client, name, max_results=10)
    print(f"Found {len(podcasts_raw)} results for '{name}'")
    return [
        podcast.PodcastMetadata(
            id=pod["id"],
            title=pod["title"],
            description=pod["description"],
            html_description=pod["htmlDescription"],
            web_url=pod["webUrl"],
        )
        for pod in podcasts_raw
    ]


@stub.function(
    image=search_image,
    shared_volumes={config.CACHE_DIR: volume},
)
def index():
    import dataclasses
    from collections import defaultdict

    import dacite

    print("Starting transcript indexing process.")
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
                ep = dacite.from_dict(data_class=podcast.EpisodeMetadata, data=data)
                episodes[ep.podcast_title].append(ep)
                guid_hash_to_episodes[ep.guid_hash] = ep

    print(f"Loaded {len(guid_hash_to_episodes)} podcast episodes.")

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

    print(f"Matched {len(search_records)} transcripts to episode records.")

    filepath = config.SEARCH_DIR / "all.json"
    print(f"writing {filepath}")
    with open(filepath, "w") as f:
        json.dump(indexed_episodes, f)

    print("calc feature vectors for all transcripts, keeping track of similar podcasts")
    X, v = search.calculate_tfidf_features(search_records)
    sim_svm = search.calculate_similarity_with_svm(X)
    filepath = config.SEARCH_DIR / "sim_tfidf_svm.json"
    print(f"writing {filepath}")
    with open(filepath, "w") as f:
        json.dump(sim_svm, f)

    print("calculate the search index to support search")
    search_dict = search.build_search_index(search_records, v)
    filepath = config.SEARCH_DIR / "search.json"
    print(f"writing {filepath}")
    with open(filepath, "w") as f:
        json.dump(search_dict, f)


@stub.function(schedule=modal.Period(hours=4))
def refresh_index():
    print(f"Running scheduled index refresh at {utc_now()}")
    index()


@web_app.post("/api/transcribe")
async def transcribe_job(podcast_id: str, episode_id: str):
    from modal import container_app

    try:
        existing_call_id = container_app.in_progress[episode_id]
        print(f"Found existing call ID {existing_call_id} for episode {episode_id}")
        return {"call_id": existing_call_id}
    except KeyError:
        pass

    call = process_episode.submit(podcast_id, episode_id)
    container_app.in_progress[episode_id] = call.object_id

    return {"call_id": call.object_id}


@web_app.get("/api/status/{call_id}")
async def poll_status(call_id: str):
    from modal._call_graph import InputInfo, InputStatus
    from modal.functions import FunctionCall

    function_call = FunctionCall.from_id(call_id)
    graph: List[InputInfo] = function_call.get_call_graph()

    try:
        map_root = graph[0].children[0].children[0]
    except IndexError:
        return dict(finished=False)

    assert map_root.function_name == "transcribe_episode"

    leaves = map_root.children
    tasks = len(set([leaf.task_id for leaf in leaves]))
    done_segments = len([leaf for leaf in leaves if leaf.status == InputStatus.SUCCESS])
    total_segments = len(leaves)
    finished = map_root.status == InputStatus.SUCCESS

    return dict(finished=finished, total_segments=total_segments, tasks=tasks, done_segments=done_segments)


def split_silences(
    path: str, min_segment_length: float = 30.0, min_silence_length: float = 1.0
) -> Iterator[Tuple[float, float]]:
    """Split audio file into contiguous chunks using the ffmpeg `silencedetect`
    filter.  Yields tuples (start, end) of each chunk in seconds."""

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

    cur_start = 0
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

    yield cur_start, duration
    num_segments += 1
    print(f"Split {path} into {num_segments} segments")


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
        model = whisper.load_model(model.name, device=device, download_root=config.MODEL_DIR)
        result = model.transcribe(f.name, language="en", fp16=use_gpu)  # , verbose=True)

    print(f"Transcribed segment {start:.2f} to {end:.2f} of {end - start:.2f} in {time.time() - t0:.2f} seconds.")

    # Add back offsets.
    for segment in result["segments"]:
        segment["start"] += start
        segment["end"] += start

    return result


@stub.function(
    image=app_image,
    shared_volumes={config.CACHE_DIR: volume},
)
def transcribe_episode(
    audio_filepath: pathlib.Path,
    result_path: pathlib.Path,
    model: config.ModelSpec,
):
    segment_gen = split_silences(str(audio_filepath))

    output_text = ""
    output_segments = []
    for result in transcribe_segment.starmap(segment_gen, kwargs=dict(audio_filepath=audio_filepath, model=model)):
        output_text += result["text"]
        output_segments += result["segments"]

    result = {"text": output_text, "segments": output_segments, "language": "en"}

    print(f"Writing openai/whisper transcription to {result_path}")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)


@stub.function(
    image=app_image,
    shared_volumes={config.CACHE_DIR: volume},
)
def process_episode(podcast_id: str, episode_id: str):
    import dacite
    import whisper
    from modal import container_app

    # pre-download the model to the cache path, because the _download fn is not
    # thread-safe.
    model = config.DEFAULT_MODEL
    whisper._download(whisper._MODELS[model.name], config.MODEL_DIR, False)

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

    print(f"Using the {model.name} model which has {model.params} parameters.")
    print(f"Wrote episode metadata to {metadata_path}")

    transcription_path = get_transcript_path(episode.guid_hash)
    if transcription_path.exists():
        print(f"Transcription already exists for '{episode.title}' with ID {episode.guid_hash}.")
        print("Skipping transcription.")
    else:
        transcribe_episode(
            audio_filepath=destination_path,
            result_path=transcription_path,
            model=model,
        )

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
    episodes_raw = podcast.fetch_episodes_data(gql, client, podcast_id, max_episodes=max_episodes)
    print(f"Retreived {len(episodes_raw)} raw episodes")
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
        if "guid" in ep
    ]
    no_guid_count = len(episodes) - len(episodes_raw)
    print(f"{no_guid_count} episodes had no GUID and couldn't be used.")
    return episodes


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "serve":
        stub.serve()
    elif cmd == "index":
        with stub.run():
            index()
    elif cmd == "search-podcast":
        with stub.run():
            for pod in search_podcast(sys.argv[2]):
                print(pod)
    else:
        exit(f"Unknown command {cmd}. Supported commands: [transcribe, run, serve, index, search-podcast]")
