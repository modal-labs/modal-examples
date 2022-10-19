import useSWR, { useSWRConfig } from "swr";
import Spinner from "../components/Spinner";
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

function ProgressBar({
  completed,
  total,
}: {
  completed: number;
  total: number;
}) {
  let percentage = Math.floor((completed / (total || 1)) * 100);
  return (
    <div className="w-full bg-gray-200 rounded-full dark:bg-gray-700 h-5 mt-4">
      {percentage > 0 && (
        <div
          className="bg-green-600 text-md font-medium text-blue-100 text-center p-0.5 leading-none rounded-full align-middle"
          style={{ width: `${percentage}%` }}
        >
          {" "}
          {percentage}%{" "}
        </div>
      )}
    </div>
  );
}

function Segment({ segment, metadata }: { segment: any; metadata: any }) {
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
              href={`${metadata.original_download_link}#t=${Math.floor(
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
              href={`${metadata.original_download_link}#t=${Math.floor(
                segment.end
              )}`}
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

interface Status {
  done_segments: number;
  total_segments: number;
  tasks: number;
}

function TranscribeProgress({
  callId,
  onFinished,
}: {
  callId: string;
  onFinished: () => void;
}) {
  const [finished, setFinished] = useState<boolean>(false);
  const [status, setStatus] = useState<Status>();
  const [intervalId, setIntervalId] = useState<number>();

  useEffect(() => {
    if (finished) {
      clearInterval(intervalId);
      return;
    }

    async function updateStatus() {
      const resp = await fetch(`/api/status/${callId}`);
      const body = await resp.json();
      setStatus(body);
      if (body.finished) {
        setFinished(true);
        onFinished();
      }
    }

    updateStatus();
    // 2s. Podcasts will take a while to transcribe.
    setIntervalId(setInterval(updateStatus, 2000));

    return () => clearInterval(intervalId);
  }, [finished]);

  let containerCount = status?.tasks ?? 0;

  return (
    <div className="flex flex-col content-center">
      <div className="flex align-center">
        <div className="flex mr-2">
          <span className="modal-barloader -rotate-[60deg]"></span>
          <span className="modal-barloader rotate-[60deg]"></span>
        </div>
        <span className="pt-1"><strong>{containerCount} Modal containers running…</strong></span>
      </div>
      <ProgressBar
        completed={status?.done_segments ?? 0}
        total={status?.total_segments ?? 1}
      />
    </div>
  );
}

function TranscribeNow({
  podcastId,
  episodeId,
  onFinished,
}: {
  podcastId: string;
  episodeId: string;
  onFinished: () => void;
}) {
  const [isTranscribing, setIsTranscribing] = useState<boolean>(false);
  const [callId, setCallId] = useState<string | null>(null);

  const transcribe = useCallback(async () => {
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

  if (isTranscribing && callId) {
    return <TranscribeProgress callId={callId} onFinished={onFinished} />;
  }

  return (
    <div className="flex flex-col content-center">
      <button
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded m-auto"
        onClick={transcribe}
        disabled={isTranscribing}
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

  const { mutate } = useSWRConfig();
  const { data } = useSWR(
    `/api/episode/${params.podcastId}/${params.episodeId}`,
    fetchData
  );

  if (!data) {
    return (
      <div className="absolute m-auto left-0 right-0 w-fit top-0 bottom-0 h-fit">
        <Spinner size={20} />
      </div>
    );
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
              onFinished={() =>
                mutate(`/api/episode/${params.podcastId}/${params.episodeId}`)
              }
            />
          )}
        </div>
      </div>

      {data.segments && (
        <div className="mx-auto max-w-4xl py-8">
          <ul className="bg-white rounded-lg border border-gray-200 w-384 text-gray-900">
            {data.segments.map((segment, idx: number) => (
              <Segment key={idx} segment={segment} metadata={data.metadata} />
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
