import { ModalWebRtcClient } from './modal_webrtc.js';

// Add status display element
const statusDisplay = document.getElementById('statusDisplay');
const MAX_STATUS_HISTORY = 100;
let statusHistory = [];


// DOM elements
const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');
const startWebcamButton = document.getElementById('startWebcamButton');
const startStreamingButton = document.getElementById('startStreamingButton');
const stopStreamingButton = document.getElementById('stopStreamingButton');

// Initialize WebRTC client
const webrtcClient = new ModalWebRtcClient();

// Set up event listeners
webrtcClient.addEventListener('status', (event) => {
    // Add timestamp to message
    const now = new Date();
    const timestamp = now.toLocaleTimeString();
    const statusLine = `[${timestamp}] ${event.detail.message}`;
    
    // Add to history
    statusHistory.push(statusLine);
    
    // Keep only last MAX_STATUS_HISTORY messages
    if (statusHistory.length > MAX_STATUS_HISTORY) {
        statusHistory.shift();
    }
    
    // Update display
    statusDisplay.innerHTML = statusHistory.map(line => 
        `<div class="status-line">${line}</div>`
    ).join('');
    
    // Scroll to bottom
    statusDisplay.scrollTop = statusDisplay.scrollHeight;
});

webrtcClient.addEventListener('localStream', (event) => {
    localVideo.srcObject = event.detail.stream;
});

webrtcClient.addEventListener('remoteStream', (event) => {
    remoteVideo.srcObject = event.detail.stream;
});

webrtcClient.addEventListener('error', (event) => {
    console.error('WebRTC error:', event.detail.error);
});

webrtcClient.addEventListener('connectionStateChange', (event) => {
    if (event.detail.state === 'connected') {
        startStreamingButton.disabled = true;
        stopStreamingButton.disabled = false;
    }
});

webrtcClient.addEventListener('streamingStopped', () => {
    stopStreamingButton.disabled = true;
    startStreamingButton.disabled = false;
    remoteVideo.srcObject = null;
});

// Initialize button states
startWebcamButton.disabled = false;
startStreamingButton.disabled = true;
stopStreamingButton.disabled = true;

// Event handlers
async function handleStartWebcam() {
    try {
        await webrtcClient.startWebcam();
        startWebcamButton.disabled = true;
        startStreamingButton.disabled = false;
    } catch (err) {
        console.error('Error starting webcam:', err);
    }
}

async function handleStartStreaming() {
    startWebcamButton.disabled = true;
    startStreamingButton.disabled = true;
    stopStreamingButton.disabled = false;
    await webrtcClient.startStreaming();
}

async function handleStopStreaming() {
    await webrtcClient.stopStreaming();
}

// Add event listener for STUN/TURN server radio buttons
document.querySelectorAll('input[name="iceServer"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        webrtcClient.setIceServerType(e.target.value);
    });
});

// Event listeners
startWebcamButton.addEventListener('click', handleStartWebcam);
startStreamingButton.addEventListener('click', handleStartStreaming);
stopStreamingButton.addEventListener('click', handleStopStreaming);

// Add cleanup handler for when browser tab is closed
window.addEventListener('beforeunload', async () => {
    await webrtcClient.cleanup();
    // ensure stun/turn radio and iceServerType are reset
    document.querySelectorAll('input[name="iceServer"]').forEach(radio => {
        if (radio.value == "turn") {
            radio.checked = false;
        } else {
            radio.checked = true;
        }
    });
    webrtcClient.setIceServerType('stun');
});
