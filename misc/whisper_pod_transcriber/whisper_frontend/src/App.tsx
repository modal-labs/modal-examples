import { useState, useEffect, useCallback } from "react";
import { HashRouter, Link, Routes, Route } from "react-router-dom";
import Podcast from "./routes/podcast";

function truncate(str: string, n: number) {
  return str.length > n ? str.slice(0, n - 1) + "…" : str;
}

function PodcastCard({ podcast }) {
  const [isSending, setIsSending] = useState(false);
  const [callId, setCallId] = useState(false);
  const [recentlyTranscribed, setRecentlyTranscribed] = useState(
    podcast.recently_transcribed === "true"
  );

  const sendRequest = useCallback(async () => {
    // don't send again while we are sending
    if (isSending) return;

    setIsSending(true);
    console.log(`Transcribing ${podcast.title} ${podcast.id}`);
    const formData = new FormData();
    formData.append("podcast_name", podcast.title);
    formData.append("podcast_id", podcast.id);

    const resp = await fetch("/api/transcribe", {
      method: "POST",
      body: formData,
    });

    if (resp.status !== 200) {
      throw new Error("An error occurred: " + resp.status);
    }
    const body = await resp.json();
    // once the request is sent, update state again
    setIsSending(false);
    setCallId(body.call_id);
  }, [isSending]); // update the callback if the state changes

  let buttonContent;
  let transcriptsLink = null;
  if (callId) {
    buttonContent = (
      <button
        className="relative text-white bg-green-500 hover:bg-green-700 font-bold py-2 px-4 rounded"
        disabled={true}
      >
        <div className="flex flex-row space-y-4">
          {recentlyTranscribed ? (
            <div className="mr-4">Waiting...</div>
          ) : (
            <div>
              <Result
                callId={callId}
                onFinished={() => {
                  setRecentlyTranscribed(true);
                  setCallId(null);
                }}
              />
            </div>
          )}
        </div>
      </button>
    );
  } else if (isSending) {
    buttonContent = (
      <button
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        disabled={true}
      >
        hi
        {/* <Spinner config={{}} /> */}
      </button>
    );
  } else if (recentlyTranscribed) {
    buttonContent = (
      <button
        className="bg-green-700 text-white font-bold py-2 px-4 rounded"
        disabled={true}
      >
        Completed
      </button>
    );
    let transcriptsHref = `/podcast/${podcast.id}`;
    transcriptsLink = (
      <Link
        to={transcriptsHref}
        className="text-blue-700 no-underline hover:underline"
      >
        <strong>View Transcripts 📃</strong>
      </Link>
    );
  } else {
    buttonContent = (
      <button
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        disabled={isSending || recentlyTranscribed}
        onClick={sendRequest}
      >
        Transcribe
      </button>
    );
  }

  return (
    <div className="max-w-2xl rounded overflow-hidden shadow-lg">
      <div className="px-6 py-4">
        <div className="font-bold text-xl mb-2">{podcast.title}</div>
        <p className="text-gray-700 text-base py-4">
          {truncate(podcast.description, 200)}
        </p>
        <div className="flow-root">
          <div className="float-left">{buttonContent}</div>
          <div className="float-right">{transcriptsLink}</div>
        </div>
      </div>
    </div>
  );
}

function PodcastList({ podcasts }) {
  const listItems = podcasts.map((pod) => (
    <li key={pod.id}>
      <PodcastCard podcast={pod} />
    </li>
  ));
  return <ul>{listItems}</ul>;
}

function Result({ callId, onFinished }) {
  const [result, setResult] = useState();
  const [intervalId, setIntervalId] = useState();
  const spinnerConfig = {
    position: "relative",
    left: "100%",
    top: "50%",
    scale: 0.5,
  };

  useEffect(() => {
    if (result) {
      clearInterval(intervalId);
      return;
    }

    const delay = 5000; // ms. Podcasts will take a while to transcribe.
    const _intervalID = setInterval(async () => {
      const resp = await fetch(`/result/${callId}`);
      if (resp.status === 200) {
        setResult(await resp.json());
        onFinished(true);
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
          {/* <Spinner config={spinnerConfig} /> */}
        </div>
      )}
    </div>
  );
}

function Form({ onSubmit }) {
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
      <div className="w-3/4 flex flex-row">
        <label>
          <span className="pr-4">
            <strong>Podcast:</strong>
          </span>
        </label>
        <input
          type="text"
          value={podcastName}
          onChange={onChange}
          placeholder="Signals and Threads podcast"
          className="flex-1 w-2/3 px-1 text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-300 cursor-pointer"
        />
      </div>
      <div>
        <button
          type="button"
          onClick={handleSubmit}
          disabled={!podcastName}
          className="bg-indigo-400 disabled:bg-zinc-500 hover:bg-indigo-600 text-white font-bold py-2 px-4 rounded text-sm"
        >
          Search
        </button>
      </div>
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
    <div className="min-w-full min-h-screen screen bg-gradient-to-r from-green-300 via-green-500 to-green-300">
      <div className="mx-auto max-w-2xl py-8">
        <main className="rounded-xl bg-white p-6">
          <Form onSubmit={handleSubmission} />
          {/* {searching && <Spinner />} */}
          {podcasts && !searching && <PodcastList podcasts={podcasts} />}
        </main>
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
      </Routes>
    </HashRouter>
  );
}

export default App;
