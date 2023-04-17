import { Link } from "react-router-dom";

export default function HomeButton() {
  return (
    <Link to="/">
      <button className="lg:fixed top-0 left-0 right-0 w-20 m-5 mb-0 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
        Home
      </button>
    </Link>
  );
}
