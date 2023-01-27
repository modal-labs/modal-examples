import pathlib

from . import config
from . import podcast
from .main import (
    stub,
    app_image,
    split_silences,
    transcribe_episode,
    transcribe_segment,
    volume,
)

logger = config.get_logger(__name__)


def _transcribe_serially(
    audio_path: pathlib.Path, offset: int = 0
) -> list[tuple[float, float]]:
    model = config.DEFAULT_MODEL
    segment_gen = split_silences(str(audio_path))
    failed_segments = []
    for i, (start, end) in enumerate(segment_gen):
        if i < offset:
            continue
        logger.info(f"Attempting transcription of ({start}, {end})...")
        try:
            transcribe_segment(
                start=start, end=end, audio_filepath=audio_path, model=model
            )
        except Exception as exc:
            logger.info(f"Transcription failed for ({start}, {end}).")
            print(exc)
            failed_segments.append((start, end))
    logger.info(f"{len(failed_segments)} failed to transcribe.")
    return failed_segments


@stub.function(
    image=app_image, shared_volumes={config.CACHE_DIR: volume}, timeout=1000
)
def test_transcribe_handles_dangling_segment():
    """
    Some podcast episodes have an empty, dangling audio segment after being split on silences.
    This test runs transcription on such an episode to check that we haven't broken transcription
    on episodes like this.

    If the transcription does fail, individual segments are checked to pull out the problem segments
    for further debugging.
    ```
    libpostproc    55.  7.100 / 55.  7.100
    [mp3 @ 0x557b828bb380] Format mp3 detected only with low score of 24, misdetection possible!
    [mp3 @ 0x557b828bb380] Failed to read frame size: Could not seek to 1026.
    /tmp/tmpuyr2iwce.mp3: Invalid argument
    ```
    """
    import ffmpeg

    # Stripped down podcast episode metadata for an episode which fails to transcribe @ commit e7093414.
    problem_episode = {
        "guid_hash": "b5b3005075fce663b3646f88a41b2b32",
        "podcast_id": "217829",
        "episode_url": "https://www.podchaser.com/podcasts/super-data-science-217829/episodes/sds-503-deep-reinforcement-lea-98045099",
        "original_download_link": "http://www.podtrac.com/pts/redirect.mp3/feeds.soundcloud.com/stream/1120216126-superdatascience-sds-503-deep-reinforcement-learning-for-robotics.mp3",
    }
    audio_path = pathlib.Path(
        config.CACHE_DIR, "test", f"{problem_episode['guid_hash']}.tmp.mp3"
    )
    audio_path.parent.mkdir(exist_ok=True)
    podcast.store_original_audio(
        url=problem_episode["original_download_link"],
        destination=audio_path,
    )

    model = config.DEFAULT_MODEL

    try:
        result_path = pathlib.Path(
            config.CACHE_DIR,
            "test",
            f"{problem_episode['guid_hash']}.transcription.json",
        )
        transcribe_episode(
            audio_filepath=audio_path,
            result_path=result_path,
            model=model,
        )
    except Exception as exc:
        print(exc)
        logger.error(
            "Transcription failed. Proceeding to checks of individual segments."
        )
    else:
        return  # Transcription worked fine.

    failed_segments = _transcribe_serially(audio_path, offset=107)
    # Checking the 1st is probably sufficient to discover bug.
    problem_segment = failed_segments[0]
    start = problem_segment[0]
    end = problem_segment[1]
    logger.info(f"Problem segment time range is ({start}, {end})")
    try:
        transcribe_segment(
            start=start, end=end, audio_filepath=audio_path, model=model
        )
    except Exception:
        logger.info(
            "Writing the problem segment to shared volume for further debugging."
        )
        bad_segment_path = pathlib.Path(
            config.CACHE_DIR,
            "test",
            f"{problem_episode['guid_hash']}.badsegment.mp3",
        )
        with open(bad_segment_path, "wb") as f:
            (
                ffmpeg.input(str(audio_path))
                .filter("atrim", start=start, end=end)
                .output(f.name)
                .overwrite_output()
                .run(quiet=True)
            )
        raise


if __name__ == "__main__":
    with stub.run():
        test_transcribe_handles_dangling_segment()
