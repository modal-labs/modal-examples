import { useState } from "react";
import { HashRouter, Link, Routes, Route } from "react-router-dom";
import Podcast from "./routes/podcast";
import Episode from "./routes/episode";
import Spinner from "./components/Spinner";
import { Search as SearchIcon } from "react-feather";

function truncate(str: string, n: number) {
  return str.length > n ? str.slice(0, n - 1) + "…" : str;
}

function PodcastCard({ podcast }) {
  return (
    <Link to={`/podcast/${podcast.id}`} className="px-6 py-1 group">
      <div className="font-bold text-xl mb-2 group-hover:underline">
        {podcast.title}
      </div>
      <p className="text-gray-700 text-base py-4">
        {truncate(podcast.description, 200)}
      </p>
    </Link>
  );
}

function PodcastList({ podcasts }) {
  const listItems = podcasts.map((pod) => (
    <li
      key={pod.id}
      className="max-w-2xl overflow-hidden border-indigo-400 border-t-2"
    >
      <PodcastCard podcast={pod} />
    </li>
  ));

  return <ul className="py-4 podcast-list">{listItems}</ul>;
}

function Form({ onSubmit, searching }) {
  const [podcastName, setPodcastName] = useState("");
  const onChange = (event) => {
    setPodcastName(event.target.value);
  };

  const handleSubmit = async (event) => {
    await onSubmit(podcastName);
  };

  return (
    <form className="flex flex-col space-y-4 items-center">
      <div className="text-2xl font-semibold text-gray-700">
        Modal Podcast Transcriber
      </div>

      <div className="w-full flex space-x-2">
        <div className="relative flex-1 w-full">
          <SearchIcon className="absolute top-[11px] left-3 w-5 h-5 text-zinc-500" />
          <input
            type="text"
            value={podcastName}
            onChange={onChange}
            placeholder="Signals and Threads podcast"
            className="h-10 w-full rounded-md pl-10 text-sm text-gray-900 bg-gray-50 border-2 border-zinc-900"
          />
        </div>
        {!searching && (
          <button
            type="submit"
            onClick={handleSubmit}
            disabled={!podcastName}
            className="bg-indigo-400 disabled:bg-zinc-500 hover:bg-indigo-600 text-white font-bold py-2 px-4 rounded text-sm w-fit"
          >
            Search
          </button>
        )}
      </div>
      <div>{searching && <Spinner size={10} />}</div>
    </form>
  );
}

function Search() {
  const [searching, setSearching] = useState(false);
  const [podcasts, setPodcasts] = useState();

  const handleSubmission = async (podcastName) => {
    const formData = new FormData();
    formData.append("podcast", podcastName);
    setSearching(true);
    const resp = await fetch("/api/podcasts", {
      method: "POST",
      body: formData,
    });

    if (resp.status !== 200) {
      throw new Error("An error occurred: " + resp.status);
    }
    const body = await resp.json();
    setPodcasts(body);
    setSearching(false);
  };

  return (
    <div className="min-w-full min-h-screen screen">
      <div className="mx-auto max-w-2xl my-8 shadow-lg rounded-xl bg-white p-6">
        <Form onSubmit={handleSubmission} searching={searching} />
        {podcasts && !searching && <PodcastList podcasts={podcasts} />}
      </div>
    </div>
  );
}

function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<Search />} />
        <Route path="podcast/:podcastId" element={<Podcast />} />
        <Route path="episode/:podcastId/:episodeId" element={<Episode />} />
      </Routes>
    </HashRouter>
  );
}

export default App;
