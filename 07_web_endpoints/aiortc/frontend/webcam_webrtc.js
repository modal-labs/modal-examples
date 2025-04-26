// Configuration
let videoProcessorUrl = null;

const RTCConfiguration = {
    iceServers: [{urls: 'stun:stun.l.google.com:19302'}],
};

async function getURL() {
    try {
        const response = await fetch('/get_url');
        const config = await response.json();
        videoProcessorUrl = config.url;
    } catch (error) {
        console.error('Failed to load url:', error);
        throw error;
    }
}

// DOM elements
const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');
const startWebcamButton = document.getElementById('startWebcamButton');
const startStreamingButton = document.getElementById('startStreamingButton');
const stopStreamingButton = document.getElementById('stopStreamingButton');

// WebRTC variables
let localStream;
let peerConnection;

// Get local media stream
async function startWebcam() {
    try {
        
        localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        localVideo.srcObject = localStream;

        startWebcamButton.disabled = true;
        startStreamingButton.disabled = false;
        
    } catch (err) {
        console.error('Error accessing media devices:', err);
    }
}

// Create and set up peer connection
async function startStreaming() {

    if (!videoProcessorUrl) {
        await getURL();
    }

    startWebcamButton.disabled = true;
    startStreamingButton.disabled = true;
    stopStreamingButton.disabled = false;

    // Create peer connection
    peerConnection = new RTCPeerConnection(RTCConfiguration);

    // Add local stream to peer connection
    localStream.getTracks().forEach(track => {
        console.log('Adding track:', track);
        peerConnection.addTrack(track, localStream);
    });

    // Handle remote stream when triggered
    peerConnection.ontrack = event => {
        console.log('Received remote stream:', event.streams[0]);
        remoteVideo.srcObject = event.streams[0];
    };

    // Handle ICE candidates using ICE trickle pattern
    // for some devices/networks, waiting for automatic ICE candidate gathering
    // to complete can take a long time
    // so we use the ICE trickle pattern to send candidates as they are gathered
    // which is much, much faster
    peerConnection.onicecandidate = async (event) => {
        if (!event.candidate) {
            return;
        }
        console.log('Sending ICE candidate:', event.candidate);
        if (event.candidate) {
            const iceCandidate = {
                candidate_sdp: event.candidate.candidate, // sdp string representation of candidate
                sdpMid: event.candidate.sdpMid,
                sdpMLineIndex: event.candidate.sdpMLineIndex,
                usernameFragment: event.candidate.usernameFragment
            };
            
            await fetch(`${videoProcessorUrl}/ice_candidate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(iceCandidate)
            });
        }
    };


    try {
        negotiate();
    } catch (err) {
        console.error('Error creating offer:', err);
    }
}

// could use websockets to do this communication
// directly and asynchrounsly - including ice candidate transfer
async function negotiate() {

    try {

        // set local description and send as offer to processor
        console.log('Setting local description...');
        await peerConnection.setLocalDescription();
        var offer = peerConnection.localDescription;
        
        console.log('Sending offer and awaiting answer...');
        const response = await fetch(`${videoProcessorUrl}/offer?` + new URLSearchParams({
            sdp: offer.sdp,
            type: offer.type
        }), {
            method: 'GET'
        });
        const answer = await response.json();

        // set remote description
        console.log('Received answer:', answer);
        console.log('Setting remote description...');

        await peerConnection.setRemoteDescription(answer);

    } catch (e) {
        alert(e);
    }
}

// Hang up the call
function stop_processing() {
    peerConnection.close();
    peerConnection = null;
    stopStreamingButton.disabled = true;
    startStreamingButton.disabled = false;
    remoteVideo.srcObject = null;
}

// Event listeners
startWebcamButton.addEventListener('click', startWebcam);
startStreamingButton.addEventListener('click', startStreaming);
stopStreamingButton.addEventListener('click', stop_processing); 