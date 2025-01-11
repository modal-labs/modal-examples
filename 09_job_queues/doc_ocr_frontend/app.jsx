const { useState, useEffect } = React;

function TextWithLineBreaks(props) {
  const textWithBreaks = props.text.split("\n").map((text, index) => (
    <React.Fragment key={index}>
      <span className="text-black">{text}</span>
      <br />
    </React.Fragment>
  ));

  return <div>{textWithBreaks}</div>;
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState("");
  const [callId, setCallId] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleExtractText = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setResult("");

    try {
      const formData = new FormData();
      formData.append("receipt", selectedFile);

      console.log(
        "Sending file:",
        selectedFile.name,
        selectedFile.type,
        selectedFile.size,
      );

      const response = await fetch("/parse", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Server response:", response.status, errorText);
        throw new Error(`Server error: ${response.status} ${errorText}`);
      }

      const data = await response.json();
      console.log("Response data:", data);
      setCallId(data.call_id);
    } catch (error) {
      console.error("Detailed error:", error);
      setResult(`Error: ${error.message}`);
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (!callId) return;

    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`/result/${callId}`);

        if (response.status === 202) {
          return;
        }

        if (response.ok) {
          const data = await response.json();
          setResult(data);
          setIsLoading(false);
          clearInterval(pollInterval);
          setCallId(null);
        } else {
          throw new Error("Failed to get results");
        }
      } catch (error) {
        console.error("Error polling results:", error);
        setResult("Error getting results");
        setIsLoading(false);
        clearInterval(pollInterval);
        setCallId(null);
      }
    }, 1000);

    return () => clearInterval(pollInterval);
  }, [callId]);

  return (
    <div className="min-h-screen bg-black text-white font-inter">
      <div className="pt-8 px-4">
        <div className="max-w-6xl mx-auto mb-8">
          <div className="bg-[#212525] rounded-3xl p-4">
            <div className="flex items-center space-x-2 px-4">
              <img
                src="images/modal_mascots.svg"
                alt="Modal Icon"
                className="w-13 h-12"
              />
              <span className="text-[#80ee64] font-light text-[20px]">
                Receipt Parsing with GOT OCR 2.0
              </span>
            </div>
          </div>
        </div>

        <div className="max-w-6xl mx-auto">
          <div className="bg-[#212525] rounded-3xl p-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              <div className="bg-[#deffdc] rounded-lg p-6 min-h-[400px] flex items-center justify-center relative">
                <div className="absolute top-0 left-0 bg-gray-100 text-gray-500 text-sm px-3 py-1 rounded-full border border-gray-200">
                  Upload a receipt
                </div>
                <div className="text-center">
                  {!previewUrl ? (
                    <>
                      <svg
                        className="w-10 h-10 relative -top-7 left-6 text-gray-500"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                        />
                      </svg>

                      <input
                        type="file"
                        onChange={handleFileChange}
                        className="hidden"
                        id="file-upload"
                        accept="image/*"
                      />
                      <label
                        htmlFor="file-upload"
                        className="cursor-pointer bg-[#80ee64] text-black px-4 py-2 rounded-full hover:opacity-90 transition-opacity font-arial inline-block font-bold"
                      >
                        Upload
                      </label>
                    </>
                  ) : (
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="max-w-full h-auto rounded-lg"
                    />
                  )}
                </div>
              </div>

              <div className="bg-[#deffdc] rounded-lg p-6 min-h-[400px] relative">
                <div className="absolute top-0 left-0 bg-[#f0f0f0] text-gray-600 text-sm px-3 py-1.5 rounded-full border border-gray-200">
                  <span>GOT-OCR Output</span>
                </div>
                <div className="bg-[#f0f0f0] rounded-lg p-4 h-full transform scale-90 overflow-auto">
                  {isLoading ? (
                    <div className="flex items-center justify-center h-full">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#80ee64]"></div>
                    </div>
                  ) : result ? (
                    <TextWithLineBreaks text={result} />
                  ) : null}
                </div>
              </div>
            </div>

            <div className="flex flex-col items-center space-y-2">
              <button
                onClick={handleExtractText}
                disabled={!selectedFile || isLoading}
                className="bg-[#80ee64] text-black px-6 py-2 rounded-full hover:opacity-90 transition-opacity font-arial font-bold disabled:opacity-50"
              >
                {isLoading ? "Processing..." : "Extract Text"}
              </button>
            </div>
          </div>
        </div>
      </div>

      <footer className="fixed bottom-0 w-full bg-[#212525] py-6">
        <a
          href="https://modal.com"
          className="flex items-center justify-end space-x-2 text-gray-400 px-8"
        >
          <span>Powered by</span>
          <img
            src="images/modal_logo.png"
            alt="Modal Logo"
            className="h-12 w-auto mx-2"
          />
        </a>
      </footer>
    </div>
  );
}

const container = document.getElementById("react");
const root = ReactDOM.createRoot(container);
root.render(React.createElement(App));
