import PulseLoader from "react-spinners/PulseLoader";

export default function Spinner({ size }: { size: number }) {
  return <PulseLoader color="rgb(79 70 229)" size={size} />;
}
