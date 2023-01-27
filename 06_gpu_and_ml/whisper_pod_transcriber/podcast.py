import dataclasses
import os
import pathlib
import urllib.request
from typing import NamedTuple, Optional, TypedDict, Union

from . import config

logger = config.get_logger(__name__)
Segment = TypedDict("Segment", {"text": str, "start": float, "end": float})


@dataclasses.dataclass
class EpisodeMetadata:
    # Unique ID of podcast this episode is associated with.
    podcast_id: Union[str, int]
    # Title of podcast this episode is associated with.
    podcast_title: Optional[str]
    title: str
    # The publish date of the episode as specified by the publisher
    publish_date: str
    # Plaintext description of episode. nb: has whitespace issues so not suitable in UI.
    description: str
    # HTML markup description. Suitable for display in UI.
    html_description: str
    # The unique identifier of this episode within the context of the podcast
    guid: str
    # Hash the guid into something appropriate for filenames.
    guid_hash: str
    # Link to episode on Podchaser website.
    episode_url: Optional[str]
    # Link to audio file for episode. Typically an .mp3 file.
    original_download_link: str


@dataclasses.dataclass
class PodcastMetadata:
    # Unique ID for a podcast
    id: str
    # Title of podcast, eg. 'The Joe Rogan Experience'.
    title: str
    # Plaintext description of episode. nb: has whitespace issues so not suitable in UI.
    description: str
    html_description: str
    # Link to podcast on Podchaser website.
    web_url: str
    # Used to detect non-English podcasts.
    language: Optional[str] = None


class DownloadResult(NamedTuple):
    data: bytes
    # Helpful to store and transmit when uploading to cloud bucket.
    content_type: str


def download_podcast_file(url: str) -> DownloadResult:
    req = urllib.request.Request(
        url,
        data=None,
        # Set a user agent to avoid 403 response from some podcast audio servers.
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
        },
    )
    with urllib.request.urlopen(req) as response:
        return DownloadResult(
            data=response.read(),
            content_type=response.headers["content-type"],
        )


def create_podchaser_client():
    """
    Use's Podchaser's graphql API to get an new access token and instantiate
    a graphql client with it.
    """
    from gql import Client, gql
    from gql.transport.aiohttp import AIOHTTPTransport

    transport = AIOHTTPTransport(url="https://api.podchaser.com/graphql")
    client = Client(transport=transport, fetch_schema_from_transport=True)
    podchaser_client_id = os.environ.get("PODCHASER_CLIENT_ID")
    podchaser_client_secret = os.environ.get("PODCHASER_CLIENT_SECRET")

    if not podchaser_client_id or not podchaser_client_secret:
        exit(
            "Must provide both PODCHASER_CLIENT_ID and PODCHASER_CLIENT_SECRET as environment vars."
        )

    query = gql(
        """
        mutation {{
            requestAccessToken(
                input: {{
                    grant_type: CLIENT_CREDENTIALS
                    client_id: "{client_id}"
                    client_secret: "{client_secret}"
                }}
            ) {{
                access_token
                token_type
            }}
        }}
    """.format(
            client_id=podchaser_client_id,
            client_secret=podchaser_client_secret,
        )
    )

    result = client.execute(query)

    access_token = result["requestAccessToken"]["access_token"]
    transport = AIOHTTPTransport(
        url="https://api.podchaser.com/graphql",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    return Client(transport=transport, fetch_schema_from_transport=True)


def search_podcast_name(gql, client, name, max_results=5) -> list[dict]:
    """
    Search for a podcast by name/title. eg. 'Joe Rogan Experience' or 'Serial'.

    This method does not paginate queries because 100s of search results is not
    useful in this application.
    """
    if max_results > 100:
        raise ValueError(
            f"A maximum of 100 results is supported, but {max_results} results were requested."
        )
    current_page = 0
    max_episodes_per_request = max_results
    search_podcast_name_query = gql(
        """
        query {{
            podcasts(searchTerm: "{name}", first: {max_episodes_per_request}, page: {current_page}) {{
                paginatorInfo {{
                    currentPage,
                    hasMorePages,
                    lastPage,
                }},
                data {{
                    id,
                    title,
                    description,
                    language,
                    htmlDescription,
                    webUrl,
                }}
            }}
        }}
        """.format(
            name=name,
            max_episodes_per_request=max_episodes_per_request,
            current_page=current_page,
        )
    )
    logger.info(f"Querying Podchaser for podcasts matching query '{name}'.")
    result = client.execute(search_podcast_name_query)
    podcasts_in_page = result["podcasts"]["data"]
    return podcasts_in_page


def fetch_episodes_data(
    gql, client, podcast_id, max_episodes=100
) -> list[dict]:
    """
    Use the Podchaser API to grab a podcast's episodes.
    """
    max_episodes_per_request = 100  # Max allowed by API
    episodes = []
    has_more_pages = True
    current_page = 0
    while has_more_pages:
        list_episodes_query = gql(
            """
            query getPodList {{
                podcast(identifier: {{id: "{id}", type: PODCHASER}}) {{
                    episodes(first: {max_episodes_per_request}, page: {current_page}) {{
                        paginatorInfo {{
                          count
                          currentPage
                          firstItem
                          hasMorePages
                          lastItem
                          lastPage
                          perPage
                          total
                        }}
                        data {{
                          id
                          title
                          airDate
                          audioUrl
                          description
                          htmlDescription
                          guid
                          url
                        }}
                    }}
                }}
            }}
        """.format(
                id=podcast_id,
                max_episodes_per_request=max_episodes_per_request,
                current_page=current_page,
            )
        )

        logger.info(f"Fetching {max_episodes_per_request} episodes from API.")
        result = client.execute(list_episodes_query)
        has_more_pages = result["podcast"]["episodes"]["paginatorInfo"][
            "hasMorePages"
        ]
        episodes_in_page = result["podcast"]["episodes"]["data"]
        episodes.extend(episodes_in_page)
        current_page += 1
        if len(episodes) >= max_episodes:
            break
    return episodes


def fetch_podcast_data(gql, client, podcast_id) -> dict:
    podcast_metadata_query = gql(
        """
        query {{
            podcast(identifier: {{id: "{podcast_id}", type: PODCHASER}}) {{
                id,
                title,
                description,
                htmlDescription,
                webUrl,
            }}
        }}
        """.format(
            podcast_id=podcast_id,
        )
    )
    logger.info(f"Querying Podchaser for podcast with ID {podcast_id}.")
    result = client.execute(podcast_metadata_query)
    return result["podcast"]


def fetch_podcast(gql, podcast_id: str) -> PodcastMetadata:
    client = create_podchaser_client()
    data = fetch_podcast_data(gql=gql, client=client, podcast_id=podcast_id)
    return PodcastMetadata(
        id=data["id"],
        title=data["title"],
        description=data["description"],
        html_description=data["htmlDescription"],
        web_url=data["webUrl"],
    )


def sizeof_fmt(num, suffix="B") -> str:
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def store_original_audio(
    url: str, destination: pathlib.Path, overwrite: bool = False
) -> None:
    if destination.exists():
        if overwrite:
            logger.info(
                f"Audio file exists at {destination} but overwrite option is specified."
            )
        else:
            logger.info(
                f"Audio file exists at {destination}, skipping download."
            )
            return

    podcast_download_result = download_podcast_file(url=url)
    humanized_bytes_str = sizeof_fmt(num=len(podcast_download_result.data))
    logger.info(f"Downloaded {humanized_bytes_str} episode from URL.")
    with open(destination, "wb") as f:
        f.write(podcast_download_result.data)
    logger.info(f"Stored audio episode at {destination}.")


def coalesce_short_transcript_segments(
    segments: list[Segment],
) -> list[Segment]:
    """
    Some extracted transcript segments from openai/whisper are really short, like even just one word.
    This function accepts a minimum segment length and combines short segments until the minimum is reached.
    """
    minimum_transcript_len = 200  # About 2 sentences.
    previous = None
    long_enough_segments = []
    for current in segments:
        if previous is None:
            previous = current
        elif len(previous["text"]) < minimum_transcript_len:
            previous = _merge_segments(left=previous, right=current)
        else:
            long_enough_segments.append(previous)
            previous = current
    if previous:
        long_enough_segments.append(previous)
    return long_enough_segments


def _merge_segments(left: Segment, right: Segment) -> Segment:
    return {
        "text": left["text"] + " " + right["text"],
        "start": left["start"],
        "end": right["end"],
    }
