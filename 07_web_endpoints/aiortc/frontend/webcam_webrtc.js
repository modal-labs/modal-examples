// Configuration
const configuration = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' }
    ]
};

// DOM elements
const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');
const startButton = document.getElementById('startButton');
const callButton = document.getElementById('callButton');
const hangupButton = document.getElementById('hangupButton');

// WebRTC variables
let localStream;
let peerConnection;

// Get local media stream
async function start() {
    try {
        localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        localVideo.srcObject = localStream;
        startButton.disabled = true;
        callButton.disabled = false;
    } catch (err) {
        console.error('Error accessing media devices:', err);
    }
}

// Create and set up peer connection
async function startProcessing() {
    callButton.disabled = true;
    hangupButton.disabled = false;


    // Create and set local description
    try {
        negotiate();
    } catch (err) {
        console.error('Error creating offer:', err);
    }
}

async function negotiate() {

    // Create peer connection
    peerConnection = new RTCPeerConnection(configuration);

    // Add local stream to peer connection
    localStream.getTracks().forEach(track => {
        console.log('Adding track:', track);
        peerConnection.addTrack(track, localStream);
    });

    // Handle remote stream
    peerConnection.ontrack = event => {
        console.log('Received remote stream:', event.streams[0]);
        remoteVideo.srcObject = event.streams[0];
    };

    try {
        const offer = await peerConnection.createOffer();
        peerConnection.setLocalDescription(offer);
        await new Promise((resolve) => {
            if (peerConnection.iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (peerConnection.iceGatheringState === 'complete') {
                        peerConnection.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                peerConnection.addEventListener('icegatheringstatechange', checkState);
            }
        });
        var offer_1 = peerConnection.localDescription;
        const response = await fetch('video-processor-url-placeholder/offer?' + new URLSearchParams({
            sdp: offer_1.sdp,
            type: offer_1.type
        }), {
            method: 'GET'
        });
        const answer = await response.json();
        console.log('Received answer:', answer);
        return peerConnection.setRemoteDescription(answer);
    } catch (e) {
        alert(e);
    }
}

// Hang up the call
function hangup() {
    peerConnection.close();
    peerConnection = null;
    hangupButton.disabled = true;
    callButton.disabled = false;
    remoteVideo.srcObject = null;
}

// Event listeners
startButton.addEventListener('click', start);
callButton.addEventListener('click', startProcessing);
hangupButton.addEventListener('click', hangup); 