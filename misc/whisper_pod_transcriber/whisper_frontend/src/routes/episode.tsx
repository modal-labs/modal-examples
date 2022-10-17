import useSWR from "swr";
import { Link } from "react-router-dom";
import { useCallback, useState, useEffect } from "react";
import { useParams } from "react-router-dom";

function formatTimestamp(total_seconds: number) {
  let milliseconds = Math.round(total_seconds * 1000.0);

  let hours = Math.floor(milliseconds / 3_600_000);
  milliseconds -= hours * 3_600_000;

  let minutes = Math.floor(milliseconds / 60_000);
  milliseconds -= minutes * 60_000;

  let seconds = Math.floor(milliseconds / 1_000);
  milliseconds -= seconds * 1_000;

  const pad = (n: number, d: number = 2) => n.toString().padStart(d, "0");

  return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}.${pad(
    milliseconds,
    3
  )}`;
}

function Segment({ segment }: { segment: any }) {
  return (
    <li className="pb-3 sm:pb-4 px-6 py-2 border-b border-gray-200 w-full rounded-t-lg">
      <div className="flex items-center space-x-4">
        <div className="flex-1 min-w-0">
          <div>{segment.text}</div>
        </div>
        <div className="inline-flex items-center text-xs bg-gray-100  text-gray-900 dark:text-white">
          <div className="hover:bg-gray-200 text-gray-800 py-1 px-1 rounded-l">
            <a
              title="listen"
              href={`${segment.episode_mp3_link}#t=${Math.floor(
                segment.start
              )}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              🎙 {formatTimestamp(segment.start)}
            </a>
          </div>
          <span className="text-gray-800 py-1 px-1">-</span>
          <div className="hover:bg-gray-200 text-gray-800 py-1 px-1 rounded-r">
            <a
              title="listen"
              href={`${segment.episode_mp3_link}#t=${Math.floor(segment.end)}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              {formatTimestamp(segment.end)}
            </a>
          </div>
        </div>
      </div>
    </li>
  );
}

function TranscribeProgress({
  callId,
  onFinished,
}: {
  callId: string;
  onFinished: () => void;
}) {
  const [result, setResult] = useState();
  const [intervalId, setIntervalId] = useState<number>();

  useEffect(() => {
    if (result) {
      clearInterval(intervalId);
      return;
    }

    const delay = 2000; // ms. Podcasts will take a while to transcribe.
    const _intervalID = setInterval(async () => {
      const resp = await fetch(`/api/transcription_status/${callId}`);
      if (resp.status === 200) {
        setResult(await resp.json());
        onFinished();
      }
    }, delay);

    setIntervalId(_intervalID);

    return () => clearInterval(intervalId);
  }, [result]);

  return (
    <div>
      {result ? (
        <span>Complete!</span>
      ) : (
        <div>
          <span>Waiting...</span>
        </div>
      )}
    </div>
  );
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
          <Link
            to={`/podcast/${params.podcastId!}`}
            className="font-bold text-l text-green-500 mb-2"
          >
            {data.metadata.podcast_title}
          </Link>
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

      {data.segments && (
        <div className="mx-auto max-w-4xl py-8">
          <ul className="bg-white rounded-lg border border-gray-200 w-384 text-gray-900">
            {data.segments.map((segment, idx: number) => (
              <Segment key={idx} segment={segment} />
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
