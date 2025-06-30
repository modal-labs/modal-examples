"""
Main web application service using FastHTML.
"""

import base64

import modal

from .common import app
from .stt import STT  # noqa: F401, to make modal deploy also deploy STT

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "python-fasthtml==0.12.20"
)


@app.function(
    scaledown_window=600,
    timeout=600,
    image=image,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    from textwrap import dedent

    import fasthtml.common as fh

    modal_logo_svg = dedent(
        """<svg width="500" height="140" viewBox="0 0 500 140" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path fill-rule="evenodd" clip-rule="evenodd" d="M58.1127 21.9828H95.5097L113.78 54.1729L77.9737 117.041H40.5705L21.8832 84.4475L58.1127 21.9828ZM59.6426 30.1081L28.1167 84.4631L42.127 108.899L73.2984 54.1683L59.6426 30.1081ZM77.9737 56.8706L46.7797 111.641H74.8348L106.029 56.8706H77.9737ZM106.037 51.4706H77.9764L64.305 27.3828H92.3654L106.037 51.4706Z" fill="#C5FFB8"/>
        <path fill-rule="evenodd" clip-rule="evenodd" d="M199.662 84.8558L180.963 117.243L143.951 116.97L107.408 54.5266L126.11 22.1345L163.681 22.2478L199.662 84.8558ZM191.86 82.1181L160.55 27.6384L132.383 27.5534L164.196 81.9143L191.86 82.1181ZM159.518 84.6121L127.682 30.2118L113.654 54.5083L145.49 108.909L159.518 84.6121ZM150.163 111.615L164.193 87.3144L191.889 87.5185L177.859 111.819L150.163 111.615Z" fill="#C5FFB8"/>
        <path d="M241.56 41.36H256.92L272.04 81.04L287.32 41.36H302.52V98H292.92V50.96H292.76L275.56 98H268.52L251.32 50.96H251.16V98H241.56V41.36ZM312.583 78.8C312.583 75.76 313.116 73.0133 314.183 70.56C315.303 68.0533 316.796 65.92 318.663 64.16C320.529 62.4 322.743 61.04 325.303 60.08C327.863 59.12 330.583 58.64 333.463 58.64C336.343 58.64 339.063 59.12 341.623 60.08C344.183 61.04 346.396 62.4 348.263 64.16C350.129 65.92 351.596 68.0533 352.663 70.56C353.783 73.0133 354.343 75.76 354.343 78.8C354.343 81.84 353.783 84.6133 352.663 87.12C351.596 89.5733 350.129 91.68 348.263 93.44C346.396 95.2 344.183 96.56 341.623 97.52C339.063 98.48 336.343 98.96 333.463 98.96C330.583 98.96 327.863 98.48 325.303 97.52C322.743 96.56 320.529 95.2 318.663 93.44C316.796 91.68 315.303 89.5733 314.183 87.12C313.116 84.6133 312.583 81.84 312.583 78.8ZM322.183 78.8C322.183 80.2933 322.423 81.7333 322.903 83.12C323.436 84.5067 324.183 85.7333 325.143 86.8C326.156 87.8667 327.356 88.72 328.743 89.36C330.129 90 331.703 90.32 333.463 90.32C335.223 90.32 336.796 90 338.183 89.36C339.569 88.72 340.743 87.8667 341.703 86.8C342.716 85.7333 343.463 84.5067 343.943 83.12C344.476 81.7333 344.743 80.2933 344.743 78.8C344.743 77.3067 344.476 75.8667 343.943 74.48C343.463 73.0933 342.716 71.8667 341.703 70.8C340.743 69.7333 339.569 68.88 338.183 68.24C336.796 67.6 335.223 67.28 333.463 67.28C331.703 67.28 330.129 67.6 328.743 68.24C327.356 68.88 326.156 69.7333 325.143 70.8C324.183 71.8667 323.436 73.0933 322.903 74.48C322.423 75.8667 322.183 77.3067 322.183 78.8ZM393.731 92.24H393.571C392.184 94.5867 390.317 96.2933 387.971 97.36C385.624 98.4267 383.091 98.96 380.371 98.96C377.384 98.96 374.717 98.4533 372.371 97.44C370.077 96.3733 368.104 94.9333 366.451 93.12C364.797 91.3067 363.544 89.1733 362.691 86.72C361.837 84.2667 361.411 81.6267 361.411 78.8C361.411 75.9733 361.864 73.3333 362.771 70.88C363.677 68.4267 364.931 66.2933 366.531 64.48C368.184 62.6667 370.157 61.2533 372.451 60.24C374.744 59.1733 377.251 58.64 379.971 58.64C381.784 58.64 383.384 58.8267 384.771 59.2C386.157 59.5733 387.384 60.0533 388.451 60.64C389.517 61.2267 390.424 61.8667 391.171 62.56C391.917 63.2 392.531 63.84 393.011 64.48H393.251V37.52H402.851V98H393.731V92.24ZM371.011 78.8C371.011 80.2933 371.251 81.7333 371.731 83.12C372.264 84.5067 373.011 85.7333 373.971 86.8C374.984 87.8667 376.184 88.72 377.571 89.36C378.957 90 380.531 90.32 382.291 90.32C384.051 90.32 385.624 90 387.011 89.36C388.397 88.72 389.571 87.8667 390.531 86.8C391.544 85.7333 392.291 84.5067 392.771 83.12C393.304 81.7333 393.571 80.2933 393.571 78.8C393.571 77.3067 393.304 75.8667 392.771 74.48C392.291 73.0933 391.544 71.8667 390.531 70.8C389.571 69.7333 388.397 68.88 387.011 68.24C385.624 67.6 384.051 67.28 382.291 67.28C380.531 67.28 378.957 67.6 377.571 68.24C376.184 68.88 374.984 69.7333 373.971 70.8C373.011 71.8667 372.264 73.0933 371.731 74.48C371.251 75.8667 371.011 77.3067 371.011 78.8ZM437.721 92.72H437.481C436.095 94.9067 434.335 96.5067 432.201 97.52C430.068 98.48 427.721 98.96 425.161 98.96C423.401 98.96 421.668 98.72 419.961 98.24C418.308 97.76 416.815 97.04 415.481 96.08C414.201 95.12 413.161 93.92 412.361 92.48C411.561 91.04 411.161 89.36 411.161 87.44C411.161 85.36 411.535 83.6 412.281 82.16C413.028 80.6667 414.015 79.44 415.241 78.48C416.521 77.4667 417.988 76.6667 419.641 76.08C421.295 75.4933 423.001 75.0667 424.761 74.8C426.575 74.48 428.388 74.2933 430.201 74.24C432.015 74.1333 433.721 74.08 435.321 74.08H437.721V73.04C437.721 70.64 436.895 68.8533 435.241 67.68C433.588 66.4533 431.481 65.84 428.921 65.84C426.895 65.84 425.001 66.2133 423.241 66.96C421.481 67.6533 419.961 68.6133 418.681 69.84L413.641 64.8C415.775 62.6133 418.255 61.04 421.081 60.08C423.961 59.12 426.921 58.64 429.961 58.64C432.681 58.64 434.975 58.96 436.841 59.6C438.708 60.1867 440.255 60.96 441.481 61.92C442.708 62.88 443.641 64 444.281 65.28C444.975 66.5067 445.455 67.76 445.721 69.04C446.041 70.32 446.228 71.5733 446.281 72.8C446.335 73.9733 446.361 75.0133 446.361 75.92V98H437.721V92.72ZM437.161 80.8H435.161C433.828 80.8 432.335 80.8533 430.681 80.96C429.028 81.0667 427.455 81.3333 425.961 81.76C424.521 82.1333 423.295 82.72 422.281 83.52C421.268 84.2667 420.761 85.3333 420.761 86.72C420.761 87.6267 420.948 88.4 421.321 89.04C421.748 89.6267 422.281 90.1333 422.921 90.56C423.561 90.9867 424.281 91.3067 425.081 91.52C425.881 91.68 426.681 91.76 427.481 91.76C430.788 91.76 433.215 90.9867 434.761 89.44C436.361 87.84 437.161 85.68 437.161 82.96V80.8ZM456.85 37.52H466.45V98H456.85V37.52Z" fill="white"/>
        </svg>"""
    )
    modal_logo_base64 = base64.b64encode(modal_logo_svg.encode()).decode()

    fast_app, rt = fh.fast_app(
        hdrs=[
            # React dependencies
            fh.Script(src="https://unpkg.com/react@18/umd/react.development.js"),
            fh.Script(
                src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"
            ),
            fh.Script(src="https://unpkg.com/@babel/standalone/babel.min.js"),
            # Audio recording libraries
            fh.Script(
                src="https://cdn.jsdelivr.net/npm/opus-recorder@latest/dist/recorder.min.js"
            ),
            fh.Script(
                src="https://cdn.jsdelivr.net/npm/opus-recorder@latest/dist/encoderWorker.min.js"
            ),
            fh.Script(
                src="https://cdn.jsdelivr.net/npm/ogg-opus-decoder/dist/ogg-opus-decoder.min.js"
            ),
            # Styling
            fh.Link(
                href="https://fonts.googleapis.com/css?family=Inter:300,400,600",
                rel="stylesheet",
            ),
            fh.Script(src="https://cdn.tailwindcss.com"),
            fh.Script("""
                tailwind.config = {
                    theme: {
                        extend: {
                            colors: {
                                ground: "#0C0F0B",
                                primary: "#9AEE86",
                                "accent-pink": "#FC9CC6",
                                "accent-blue": "#B8E4FF",
                            },
                        },
                    },
                };
            """),
        ],
    )

    app_js = dedent(
        """
        const { useRef, useEffect, useState } = React;

        const getBaseURL = () => {
            const currentURL = new URL(window.location.href);
            let hostname = currentURL.hostname;
            hostname = hostname.replace('-web', '-stt-web');
            const wsProtocol = currentURL.protocol === 'https:' ? 'wss:' : 'ws:';
            return `${wsProtocol}//${hostname}/ws`;
        }

        const App = () => {
            // Mic Input
            const [recorder, setRecorder] = useState(null);
            const [amplitude, setAmplitude] = useState(0);

            // WebSocket
            const socketRef = useRef(null);

            // UI State
            const [warmupComplete, setWarmupComplete] = useState(false);
            const [completedSentences, setCompletedSentences] = useState([]);
            const [pendingSentence, setPendingSentence] = useState('');

            // Mic Input: start the Opus recorder
            const startRecording = async () => {
                // prompts user for permission to use microphone
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

                const recorder = new Recorder({
                    encoderPath: "https://cdn.jsdelivr.net/npm/opus-recorder@latest/dist/encoderWorker.min.js",
                    streamPages: true,
                    encoderApplication: 2049,
                    encoderFrameSize: 80,  // milliseconds, equal to 1920 samples at 24000 Hz
                    encoderSampleRate: 24000,  // 24000 to match model's sample rate
                    maxFramesPerPage: 1,
                    numberOfChannels: 1,
                });

                recorder.ondataavailable = async (arrayBuffer) => {
                    if (socketRef.current) {
                        if (socketRef.current.readyState !== WebSocket.OPEN) {
                            console.log("Socket not open, dropping audio");
                            return;
                        }
                        await socketRef.current.send(arrayBuffer);
                    }
                };

                recorder.start().then(() => {
                    console.log("Recording started");
                    setRecorder(recorder);
                });

                // create a MediaRecorder object for capturing PCM (calculating amplitude)
                const analyzerContext = new (window.AudioContext || window.webkitAudioContext)();
                const analyzer = analyzerContext.createAnalyser();
                analyzer.fftSize = 256;
                const sourceNode = analyzerContext.createMediaStreamSource(stream);
                sourceNode.connect(analyzer);

                // Use a separate audio processing function instead of MediaRecorder
                const processAudio = () => {
                    const dataArray = new Uint8Array(analyzer.frequencyBinCount);
                    analyzer.getByteFrequencyData(dataArray);
                    const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
                    setAmplitude(average);
                    requestAnimationFrame(processAudio);
                };
                processAudio();
            };

        // WebSocket: open websocket connection and start recording
        useEffect(() => {
            const endpoint = getBaseURL();
            console.log("Connecting to", endpoint);
            const socket = new WebSocket(endpoint);
            socketRef.current = socket;

            socket.onopen = () => {
                console.log("WebSocket connection opened");
                startRecording();
                setWarmupComplete(true);
            };

            socket.onmessage = async (event) => {
                // data is a blob, convert to array buffer
                const arrayBuffer = await event.data.arrayBuffer();
                const view = new Uint8Array(arrayBuffer);
                const tag = view[0];
                const payload = arrayBuffer.slice(1);
                if (tag === 2) {
                    // text data
                    const decoder = new TextDecoder();
                    const text = decoder.decode(payload);

                    setPendingSentence(prevPending => {
                        const updatedPending = prevPending + text;
                        if (updatedPending.endsWith('.') || updatedPending.endsWith('!') || updatedPending.endsWith('?')) {
                            setCompletedSentences(prevCompleted => [...prevCompleted, updatedPending]);
                            return '';
                        }
                        return updatedPending;
                    });
                }
            };

            socket.onclose = () => {
                console.log("WebSocket connection closed");
            };

            return () => {
                socket.close();
            };
        }, []);

        return React.createElement(React.Fragment, null,
            React.createElement(AudioControl, { recorder, amplitude }),
            React.createElement(TextOutput, { warmupComplete, completedSentences, pendingSentence })
        );
        }

        const AudioControl = ({ recorder, amplitude }) => {
            const [muted, setMuted] = useState(true);

            const toggleMute = () => {
                if (!recorder) return;
                setMuted(!muted);
                recorder.setRecordingGain(muted ? 1 : 0);
            };

            // unmute automatically once the recorder is ready
            useEffect(() => {
                if (recorder) {
                setMuted(false);
                recorder.setRecordingGain(1);
                }
            }, [recorder]);

            const amplitudePercent = amplitude / 255;
            const maxAmplitude = 0.3; // for scaling
            const minDiameter = 30; // minimum diameter of the circle in pixels
            const maxDiameter = 200; // increased maximum diameter to ensure overflow

            var diameter = minDiameter + (maxDiameter - minDiameter) * (amplitudePercent / maxAmplitude);
            if (muted) diameter = 20;

            const audioEl = document.getElementById('audio-control');
            if (audioEl) {
                audioEl.innerHTML = `
                <div class="w-full h-6 rounded-sm relative overflow-hidden">
                    <div class="absolute inset-0 flex items-center justify-center">
                    <div
                        class="rounded-full transition-all duration-100 ease-out hover:cursor-pointer ${muted ? 'bg-gray-200 hover:bg-red-300' : 'bg-red-500 hover:bg-red-300'}"
                        onclick="window.toggleMute && window.toggleMute()"
                        style="width: ${diameter}px; height: ${diameter}px;"
                    ></div>
                    </div>
                </div>
                `;
            }

            window.toggleMute = toggleMute;

            return null;
        };

        const TextOutput = ({ warmupComplete, completedSentences, pendingSentence }) => {
            const containerRef = useRef(null);
            const allSentences = [...completedSentences, pendingSentence];
            if (pendingSentence.length === 0 && allSentences.length > 1) {
                allSentences.pop();
            }

        useEffect(() => {
            const container = document.getElementById('text-output');
            if (container) {
                if (warmupComplete) {
                    container.innerHTML = allSentences.map((sentence, index) =>
                    `<p class="text-gray-300 my-2">${sentence}</p>`
                    ).reverse().join('');
                } else {
                    container.innerHTML = '<p class="text-gray-400 animate-pulse">Warming up model...</p>';
                }
                container.scrollTop = container.scrollHeight;
            }
        }, [completedSentences, pendingSentence, warmupComplete]);

        return null;
        };

        const container = document.getElementById("react-app");
        if (container) {
            ReactDOM.createRoot(container).render(React.createElement(App));
        }
        """
    )

    @rt("/")
    def get():
        return (
            fh.Title(
                "Kyutai STT",
            ),
            fh.Body(
                # Main container
                fh.Div(
                    # Card container
                    fh.Div(
                        fh.Div(
                            fh.Div(
                                id="text-output",
                                cls="flex flex-col-reverse overflow-y-auto max-h-64 pr-2",
                            ),
                            cls="w-5/6 overflow-y-auto max-h-64",
                        ),
                        # Audio control area
                        fh.Div(
                            fh.Div(
                                id="audio-control",
                                cls="w-full h-full flex items-center",
                            ),
                            cls="w-1/6 ml-4 pl-4",
                        ),
                        cls="flex",
                    ),
                    cls="bg-gray-800 rounded-lg shadow-lg w-full max-w-xl p-6 mb-8",
                ),
                fh.Footer(
                    fh.Span(
                        "Built with ",
                        fh.A(
                            "Kyutai STT",
                            href="https://github.com/kyutai-labs/delayed-streams-modeling",
                            target="_blank",
                            rel="noopener noreferrer",
                            cls="underline",
                        ),
                        " and",
                        cls="text-sm font-medium text-gray-300 mr-2",
                    ),
                    fh.A(
                        fh.Img(
                            src=f"data:image/svg+xml;base64,{modal_logo_base64}",
                            alt="Modal logo",
                            cls="w-24",
                        ),
                        cls="flex items-center px-3 py-2 rounded-lg bg-gray-800 shadow-lg hover:bg-gray-700 transition-colors duration-200",
                        href="https://modal.com",
                        target="_blank",
                        rel="noopener noreferrer",
                    ),
                    cls="fixed bottom-4 inline-flex items-center justify-center",
                ),
                fh.Div(id="react-app"),
                fh.Script(app_js, type="text/babel"),
                cls="relative bg-gray-900 text-white min-h-screen flex flex-col items-center justify-center p-4",
            ),
        )

    return fast_app
