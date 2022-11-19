import { Link } from "react-router-dom";
import modalLogImg from '../modal-logo.svg'

export default function Footer() {
  return (
        <div
            className="fixed bottom-0 content-center flex justify-center" 
            style={{
                width: "100%",
                color: "white",
                textAlign: "center",
                zIndex: 100,
              }}>
            <a 
                href="https://modal.com"
                target="_blank"
                rel="noopener noreferrer"
            >
                <footer className="flex flex-row items-center w-42 p-1 bg-zinc-800 mb-6 rounded shadow-lg">
                    <span className="p-1 text-md"><strong>built with</strong></span>
                    <img className="h-12 w-24" src={modalLogImg}></img>
                </footer>
            </a>
        </div>
  );
}
