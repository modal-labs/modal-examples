import useSWR from "swr";
import { useCallback, useState } from "react";
import { useParams } from "react-router-dom";

function Segment({
  title,
  publishDate,
}: {
  title: string;
  publishDate: string;
}) {
  return "foo";
  // <li className="px-6 py-2 border-b border-gray-200 w-full rounded-t-lg" >
  //   <Link
  //     to="/episode/{ep.podcast_id}/{ep.guid_hash}"
  //     className="text-blue-700 no-underline hover:underline"
  //   >
  //     {title}
  //   </Link>{" "}
  //   | {publishDate}
  // </li>
}

function TranscribeNow({
  podcastId,
  episodeId,
}: {
  podcastId: string;
  episodeId: string;
}) {
  const [isTranscribing, setIsTranscribing] = useState<boolean>(false);
  const [callId, setCallId] = useState<string | null>(null);

  const transcribe = useCallback(async () => {
    if (isTranscribing) {
      return;
    }

    setIsTranscribing(true);

    const resp = await fetch(
      "/api/transcribe?" +
        new URLSearchParams({ podcast_id: podcastId, episode_id: episodeId }),
      { method: "POST" }
    );

    if (resp.status !== 200) {
      throw new Error("An error occurred: " + resp.status);
    }

    const body = await resp.json();
    setCallId(body.call_id);
  }, [isTranscribing]);

  return (
    <div className="flex flex-col content-center">
      <button
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded m-auto"
        onClick={transcribe}
      >
        Transcribe Now
      </button>
    </div>
  );
}

export default function Podcast() {
  let params = useParams();

  async function fetchData() {
    const response = await fetch(
      `/api/episode/${params.podcastId}/${params.episodeId}`
    );
    const data = await response.json();
    return data;
  }

  const { data } = useSWR(
    `/api/episode/${params.podcastId}/${params.episodeId}`,
    fetchData
  );

  if (!data) {
    return <div>Loading...</div>;
  }

  return (
    <div className="flex flex-col">
      <div className="mx-auto max-w-4xl mt-4 py-8 rounded overflow-hidden shadow-lg">
        <div className="px-6 py-4">
          <div className="font-bold text-l text-green-500 mb-2">
            {data.metadata.podcast_title}
          </div>
          <div className="font-bold text-xl mb-2">{data.metadata.title}</div>
          <div className="text-gray-700 text-sm py-4">
            {data.metadata.description}
          </div>
          {!data.segments && (
            <TranscribeNow
              podcastId={params.podcastId!}
              episodeId={params.episodeId!}
            />
          )}
        </div>
      </div>

      <div className="mx-auto max-w-4xl py-8">
        {
          data.segments && <div>foo</div>
          /* <ul className="bg-white rounded-lg border border-gray-200 w-384 text-gray-900">
                              {.episodes.map((ep) => (
                                  <Epsiode
                                      key={ep.guid_hash}
                                      title={ep.title}
                                      publishDate={ep.publish_date}
                                  />
                              ))}
                          </ul> */
        }
      </div>
    </div>
  );
}
