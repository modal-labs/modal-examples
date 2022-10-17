import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";

function Epsiode({
  key,
  title,
  publishDate,
}: {
  key: string;
  title: string;
  publishDate: string;
}) {
  return (
    <li
      key={key}
      className="px-6 py-2 border-b border-gray-200 w-full rounded-t-lg"
    >
      <a
        href="/transcripts/{ep.podcast_id}/{ep.guid_hash}"
        className="text-blue-700 no-underline hover:underline"
      >
        {title}
      </a>{" "}
      | {publishDate}
    </li>
  );
}

export default function Podcast() {
  let params = useParams();
  let [podcastInfo, setPodcastInfo] = useState(null);
  useEffect(() => {
    const fetchData = async () => {
      const resp = await fetch(`/api/podcast/${params.podcastId}`);
      const body = await resp.json();
      setPodcastInfo(body);
    };

    fetchData().catch(console.error);
  }, []);

  if (!podcastInfo) {
    return <div>Loading...</div>;
  }

  return (
    <div>
      <div className="mx-auto max-w-4xl mt-4 py-8 rounded overflow-hidden shadow-lg">
        <div className="px-6 py-4">
          <div className="font-bold text-xl">
            {podcastInfo.pod_metadata.title}
          </div>
          <div className="text-gray-700 text-md">
            {podcastInfo.pod_metadata.description}
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-4xl py-8">
        <ul className="bg-white rounded-lg border border-gray-200 w-384 text-gray-900">
          {podcastInfo.episodes.map((ep) => (
            <Epsiode
              key={ep.guid_hash}
              title={ep.title}
              publishDate={ep.publish_date}
            />
          ))}
        </ul>
      </div>
    </div>
  );
}
