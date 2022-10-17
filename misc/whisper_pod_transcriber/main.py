"""
whisper-pod-transcriber uses OpenAI's Whisper modal to do speech-to-text transcription
of podcasts.
"""
import dataclasses
import datetime
import json
import pathlib
import sys
from typing import Iterator, Tuple

import modal
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from . import config, podcast, search, web

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

stub = modal.Stub("whisper-pod-transcriber", image=app_image)
web_app = FastAPI()


def utc_now() -> datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def transcript_path(guid_hash: str) -> pathlib.Path:
    return config.TRANSCRIPTIONS_DIR / f"{guid_hash}.json"


@web_app.get("/api/all")
async def all_transcripts():
    from collections import defaultdict

    import dacite

    episodes_by_show = defaultdict(list)
    if config.METADATA_DIR.exists():
        for file in config.METADATA_DIR.iterdir():
            with open(file, "r") as f:
                data = json.load(f)
                ep = dacite.from_dict(data_class=podcast.EpisodeMetadata, data=data)
                episodes_by_show[ep.podcast_title].append(ep)

    body = web.html_all_transcripts_header()
    for show, episodes_by_show in episodes_by_show.items():
        episode_part = f"""<div class="font-bold text-center text-green-500 text-xl mt-6">{show}</div>"""
        episode_part += web.html_episode_list(episodes_by_show)
        body += episode_part
    content = web.html_page(title="Modal Podcast Transcriber | All Transcripts", body=body)
    return HTMLResponse(content=content, status_code=200)


@web_app.get("/api/transcripts/{podcast_id}/{episode_guid_hash}")
async def episode_transcript_page(podcast_id: str, episode_guid_hash):
    import dacite

    episode_metadata_path = config.METADATA_DIR / f"{episode_guid_hash}.json"
    transcription_path = transcript_path(episode_guid_hash)
    with open(transcription_path, "r") as f:
        data = json.load(f)

    with open(episode_metadata_path, "r") as f:
        metadata = json.load(f)
        episode = dacite.from_dict(data_class=podcast.EpisodeMetadata, data=metadata)

    segments_ul_html = web.html_transcript_list(data["segments"], episode_mp3_link=episode.original_download_link)
    episode_header_html = web.html_episode_header(episode)
    body = episode_header_html + segments_ul_html
    content = web.html_page(title="Modal Podcast Transcriber | Episode Transcript", body=body)
    return HTMLResponse(content=content, status_code=200)


@stub.function(secret=modal.Secret.from_name("podchaser"), shared_volumes={config.CACHE_DIR: volume})
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
        metadata_path = metadata_dir / f"{ep.guid_hash}.json"
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
            ep["transcribed"] = transcript_path(ep["guid_hash"]).exists()
            episodes.append(ep)

    episodes.sort(key=lambda ep: ep.get("publish_date"), reverse=True)

    return JSONResponse(content={"pod_metadata": pod_metadata, "episodes": episodes})


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
    return JSONResponse(content=podcasts_response)


@stub.asgi(
    mounts=[modal.Mount("/assets", local_dir=config.ASSETS_PATH)],
    shared_volumes={config.CACHE_DIR: volume},
)
def fastapi_app():
    import fastapi.staticfiles

    web_app.mount("/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True))

    return web_app


@stub.function(schedule=modal.Period(hours=4))
def refresh_index():
    print(f"Running scheduled index refresh at {utc_now()}")
    index()


@stub.function(
    image=app_image,
    secret=modal.Secret.from_name("podchaser"),
)
def search_podcast(name):
    from gql import gql

    print(f"Searching for '{name}'")
    client = podcast.create_podchaser_client()
    podcasts_raw = podcast.search_podcast_name(gql, client, name, max_results=3)
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
    if config.METADATA_DIR.exists():
        for file in config.METADATA_DIR.iterdir():
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

    filepath = pathlib.Path(config.SEARCH_DIR, "jall.json")
    print(f"writing {filepath}")
    with open(filepath, "w") as f:
        json.dump(indexed_episodes, f)

    print("calc feature vectors for all transcripts, keeping track of similar podcasts")
    X, v = search.calculate_tfidf_features(search_records)
    sim_svm = search.calculate_similarity_with_svm(X)
    filepath = pathlib.Path(config.SEARCH_DIR, "sim_tfidf_svm.json")
    print(f"writing {filepath}")
    with open(filepath, "w") as f:
        json.dump(sim_svm, f)

    print("calculate the search index to support search")
    search_dict = search.build_search_index(search_records, v)
    filepath = pathlib.Path(config.SEARCH_DIR, "search.json")
    print(f"writing {filepath}")
    with open(filepath, "w") as f:
        json.dump(search_dict, f)


@web_app.post("/api/transcribe")
async def transcribe_job(request: Request):
    form = await request.form()
    pod_id = form["podcast_id"]
    call = transcribe_podcast.submit(podcast_id=pod_id)
    return {"call_id": call.object_id}


@web_app.get("/api/result/{call_id}")
async def poll_results(call_id: str):
    from modal.functions import FunctionCall

    function_call = FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=0)
    except TimeoutError:
        return JSONResponse(status_code=202)

    return result


@stub.function(
    image=app_image,
    shared_volumes={config.CACHE_DIR: volume},
    secret=modal.Secret.from_name("podchaser"),
    concurrency_limit=2,
)
def transcribe_podcast(podcast_id: str, model: config.ModelSpec = config.DEFAULT_MODEL):
    # pre-download the model to the cache path, because the _download fn is not
    # thread-safe.
    import whisper
    from gql import gql

    whisper._download(whisper._MODELS[model.name], config.MODEL_DIR, False)

    pod_metadata: podcast.PodcastMetadata = podcast.fetch_podcast(gql, podcast_id)

    config.PODCAST_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = config.PODCAST_METADATA_DIR / f"{podcast_id}.json"
    with open(metadata_path, "w") as f:
        json.dump(dataclasses.asdict(pod_metadata), f)
    print(f"Wrote podcast metadata to {metadata_path}")

    temp_limit = config.transcripts_per_podcast_limit
    print(f"Fetching {temp_limit} podcast episodes to transcribe.")
    episodes = fetch_episodes(show_name=pod_metadata.title, podcast_id=podcast_id)
    # Most recent episodes
    episodes.sort(key=lambda ep: ep.publish_date, reverse=True)
    completed = []
    for result in process_episode.map(episodes[:temp_limit], order_outputs=False):
        print(f"Processed episode '{result.title}'")
        completed.append(result.title)

    config.COMPLETED_DIR.mkdir(parents=True, exist_ok=True)
    completion_marker_path = config.COMPLETED_DIR / f"{podcast_id}.txt"
    with open(completion_marker_path, "w") as f:
        f.write(str(utc_now()))
    print(f"Marked podcast {podcast_id} as recently transcribed.")
    return completed  # Need to return something for function call polling to work.


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
def process_episode(episode: podcast.EpisodeMetadata):
    config.RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    config.TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    destination_path = config.RAW_AUDIO_DIR / episode.guid_hash
    podcast.store_original_audio(
        url=episode.original_download_link,
        destination=destination_path,
    )

    model = config.supported_whisper_models["base.en"]
    print(f"Using the {model.name} model which has {model.params} parameters.")

    metadata_path = config.METADATA_DIR / f"{episode.guid_hash}.json"
    with open(metadata_path, "w") as f:
        json.dump(dataclasses.asdict(episode), f)
    print(f"Wrote episode metadata to {metadata_path}")

    transcription_path = transcript_path(episode.guid_hash, model)
    # if transcription_path.exists():
    #     print(f"Transcription already exists for '{episode.title}' with ID {episode.guid_hash}.")
    #     print("Skipping GPU transcription.")
    # else:
    transcribe_episode(
        audio_filepath=destination_path,
        result_path=transcription_path,
        model=model,
    )
    return episode


@stub.function(
    image=app_image,
    secret=modal.Secret.from_name("podchaser"),
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
    if cmd == "transcribe":
        podcast_id = sys.argv[2]
        with stub.run() as app:
            print(f"Modal app ID -> {app.app_id}")
            transcribe_podcast(podcast_id=podcast_id)
    elif cmd == "serve":
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
