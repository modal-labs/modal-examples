let socket = null;
let audioContext = null;
let audioQueue = [];
let isPlaying = false;
let shouldStop = false;
let sourceNode = null;
let modelReady = false;

const textInput = document.getElementById('text-input');
const voiceSelect = document.getElementById('voice-select');
const synthesizeBtn = document.getElementById('synthesize-btn');
const stopBtn = document.getElementById('stop-btn');
const statusDiv = document.getElementById('status');

const getBaseURL = () => {
    const currentURL = new URL(window.location.href);
    let hostname = currentURL.hostname;
    hostname = hostname.replace('-web', '-tts-web');
    const wsProtocol = currentURL.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${wsProtocol}//${hostname}/ws`;
};

const updateStatus = (message, isLoading = false) => {
    statusDiv.textContent = message;
    if (isLoading) {
        statusDiv.classList.add('animate-pulse');
    } else {
        statusDiv.classList.remove('animate-pulse');
    }
};

const initAudioContext = async () => {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 24000
        });
    }
    
    if (audioContext.state === 'suspended') {
        await audioContext.resume();
    }
};

const playAudioChunk = async (audioData) => {
    if (shouldStop) return;
    
    try {
        const int16Array = new Int16Array(audioData);
        
        const audioBuffer = audioContext.createBuffer(1, int16Array.length, audioContext.sampleRate);
        const channelData = audioBuffer.getChannelData(0);
        
        for (let i = 0; i < int16Array.length; i++) {
            channelData[i] = int16Array[i] / 32768.0;
        }
        
        sourceNode = audioContext.createBufferSource();
        sourceNode.buffer = audioBuffer;
        sourceNode.connect(audioContext.destination);
        
        return new Promise((resolve) => {
            sourceNode.onended = resolve;
            sourceNode.start();
        });
    } catch (error) {
        console.error('Error playing audio chunk:', error);
    }
};

const processAudioQueue = async () => {
    isPlaying = true;
    
    while (audioQueue.length > 0 && !shouldStop) {
        const audioChunk = audioQueue.shift();
        await playAudioChunk(audioChunk);
    }
    
    isPlaying = false;
    
    if (shouldStop) {
        updateStatus('Playback stopped');
    } else {
        updateStatus('Playback complete');
    }
    
    synthesizeBtn.disabled = false;
    stopBtn.disabled = true;
};

const connectWebSocket = () => {
    const endpoint = getBaseURL();
    console.log("Connecting to", endpoint);
    
    socket = new WebSocket(endpoint);
    socket.binaryType = 'arraybuffer';
    
    socket.onopen = () => {
        console.log("WebSocket connection opened");
        synthesizeBtn.disabled = false;
        modelReady = true;
        updateStatus('Ready');
    };
    
    socket.onmessage = async (event) => {
        const dataView = new DataView(event.data);
        const tag = dataView.getUint8(0);
        
        if (tag === 1) {
            const audioData = event.data.slice(1);
            audioQueue.push(audioData);
            
            if (!isPlaying && !shouldStop) {
                processAudioQueue();
            }
        }
    };
    
    socket.onclose = () => {
        console.log("WebSocket connection closed");
        updateStatus('Disconnected from server');
        socket = null;
        modelReady = false;
        
        synthesizeBtn.disabled = true;
        stopBtn.disabled = true;
    };
    
    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus('Connection error');
    };
};

const synthesize = async () => {
    const text = textInput.value.trim();
    const voice = voiceSelect.value;
    
    if (!text) {
        updateStatus('Please enter some text');
        return;
    }
    
    await initAudioContext();
    
    audioQueue = [];
    shouldStop = false;
    
    synthesizeBtn.disabled = true;
    stopBtn.disabled = false;
    updateStatus('Synthesizing...', true);
    
    socket.send(JSON.stringify({
        text: text,
        voice: voice
    }));
};

const stop = () => {
    shouldStop = true;
    audioQueue = [];
    
    if (sourceNode) {
        sourceNode.stop();
        sourceNode = null;
    }
    
    synthesizeBtn.disabled = false;
    stopBtn.disabled = true;
    updateStatus('Stopped');
};

synthesizeBtn.addEventListener('click', synthesize);
stopBtn.addEventListener('click', stop);

textInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        if (!synthesizeBtn.disabled) {
            synthesize();
        }
    }
});

updateStatus('Connecting...', true);

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', connectWebSocket);
} else {
    connectWebSocket();
}