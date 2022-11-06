import json
from typing import List

from fastapi import FastAPI, Request

from . import config
from .main import (
    get_episode_metadata_path,
    get_transcript_path,
    populate_podcast_metadata,
    process_episode,
    search_podcast,
)
from .podcast import coalesce_short_transcript_segments

web_app = FastAPI()


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

    return dict(metadata=metadata, segments=coalesce_short_transcript_segments(data["segments"]))


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


@web_app.post("/api/podcasts")
async def podcasts_endpoint(request: Request):
    import dataclasses

    form = await request.form()
    name = form["podcast"]
    podcasts_response = []
    for pod in search_podcast(name):
        podcasts_response.append(dataclasses.asdict(pod))
    return podcasts_response


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
        function_call.get(timeout=0.1)
    except TimeoutError:
        pass
    except Exception as exc:
        if exc.args:
            inner_exc = exc.args[0]
            if "HTTPError 403" in inner_exc:
                return dict(error="permission denied on podcast audio download")
        return dict(error="unknown job processing error")

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
