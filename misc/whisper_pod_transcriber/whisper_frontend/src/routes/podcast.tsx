import useSWR from "swr";
import { useParams } from "react-router-dom";
import { Link } from "react-router-dom";
import HomeButton from "../components/HomeButton";
import Spinner from "../components/Spinner";

function Episode({
  guidHash,
  title,
  transcribed,
  publishDate,
  podcastId,
}: {
  guidHash: string;
  title: string;
  transcribed: boolean;
  publishDate: string;
  podcastId: string;
}) {
  return (
    <li
      key={guidHash}
      className="px-6 py-2 border-b border-gray-200 w-full rounded-t-lg"
    >
      {transcribed ? "📃 " : "  "}
      <Link
        to={`/episode/${podcastId}/${guidHash}`}
        className="text-blue-700 no-underline hover:underline"
      >
        {title}
      </Link>{" "}
      | {publishDate}
    </li>
  );
}

export default function Podcast() {
  let params = useParams();

  async function fetchData() {
    const response = await fetch(`/api/podcast/${params.podcastId}`);
    const data = await response.json();
    return data;
  }

  const { data } = useSWR(`/api/podcast/${params.podcastId}`, fetchData);

  if (!data) {
    return (
      <div className="absolute m-auto left-0 right-0 w-fit top-0 bottom-0 h-fit">
        <Spinner size={20} />
      </div>
    );
  }

  return (
    <div className="w-full">
      <div>
        <HomeButton />
        <div className="mx-auto max-w-4xl mt-4 py-8 rounded overflow-hidden shadow-lg">
          <div className="px-6 py-4">
            <div className="font-bold text-xl">{data.pod_metadata.title}</div>
            <div className="text-gray-700 text-md py-1">
              {data.pod_metadata.description}
            </div>
          </div>
        </div>

        <div className="mx-auto max-w-4xl py-8">
          <ul className="bg-white rounded-lg border border-gray-200 w-384 text-gray-900">
            {data.episodes.map((ep) => (
              <Episode
                key={ep.guid_hash}
                transcribed={ep.transcribed}
                guidHash={ep.guid_hash}
                title={ep.title}
                publishDate={ep.publish_date}
                podcastId={params.podcastId!}
              />
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
