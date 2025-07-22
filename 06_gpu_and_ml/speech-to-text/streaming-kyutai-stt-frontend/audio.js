
let recorder = null;
let socket = null;
let warmupComplete = false;
let completedSentences = [];
let pendingSentence = '';

const getBaseURL = () => {
    const currentURL = new URL(window.location.href);
    let hostname = currentURL.hostname;
    hostname = hostname.replace('-ui', '-stt-api');
    const wsProtocol = currentURL.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${wsProtocol}//${hostname}/ws`;
}

const updateTextOutput = () => {
    const container = document.getElementById('text-output');
    if (!container) return;

    const allSentences = [...completedSentences];
    if (pendingSentence) {
        allSentences.push(pendingSentence);
    }

    let content = '';

    if (warmupComplete) {
        content += allSentences.map(sentence =>
            `<p class="text-gray-300 my-2">${sentence}</p>`
        ).reverse().join('');
    } else {
        content = '<p class="text-gray-400 animate-pulse">Warming up model...</p>';
    }

    container.innerHTML = content;
    container.scrollTop = container.scrollHeight;
};

const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    recorder = new Recorder({
        encoderPath: "https://cdn.jsdelivr.net/npm/opus-recorder@latest/dist/encoderWorker.min.js",
        streamPages: true,
        encoderApplication: 2049,
        encoderFrameSize: 80,
        encoderSampleRate: 24000,
        maxFramesPerPage: 1,
        numberOfChannels: 1,
    });

    recorder.ondataavailable = async (arrayBuffer) => {
        if (socket && socket.readyState === WebSocket.OPEN) {
            await socket.send(arrayBuffer);
        }
    };

    recorder.start().then(() => {
        console.log("Recording started");
        recorder.setRecordingGain(1);
    });

    const analyzerContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyzer = analyzerContext.createAnalyser();
    analyzer.fftSize = 256;
    const sourceNode = analyzerContext.createMediaStreamSource(stream);
    sourceNode.connect(analyzer);

    const processAudio = () => {
        const dataArray = new Uint8Array(analyzer.frequencyBinCount);
        analyzer.getByteFrequencyData(dataArray);
        requestAnimationFrame(processAudio);
    };
    processAudio();
};

const initApp = () => {
    const endpoint = getBaseURL();
    console.log("Connecting to", endpoint);
    socket = new WebSocket(endpoint);

    socket.onopen = () => {
        console.log("WebSocket connection opened");
        startRecording();
        warmupComplete = true;
        updateTextOutput();
    };

    socket.onmessage = async (event) => {
        // data is a blob, convert to array buffer
        const arrayBuffer = await event.data.arrayBuffer();
        const view = new Uint8Array(arrayBuffer);
        const tag = view[0];
        const payload = arrayBuffer.slice(1);

        if (tag === 1) {
            // text data
            const decoder = new TextDecoder();
            const text = decoder.decode(payload);
            pendingSentence += text;
        }

        if (pendingSentence.endsWith('.') || pendingSentence.endsWith('!') || pendingSentence.endsWith('?')) {
            completedSentences.push(pendingSentence);
            pendingSentence = '';
        }

        updateTextOutput();
    };

    socket.onclose = () => {
        console.log("WebSocket connection closed");
    };

    updateTextOutput();
};

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
