// Configuration
let config = null;

const RTCConfiguration = {
    // sdpSemantics: 'unified-plan', //newer implementation of WebRTC
    iceServers: [{urls: 'stun:stun.l.google.com:19302'}],
    // iceCandidatePoolSize: 1
};

// Initialize configuration
async function initConfig() {
    try {
        const response = await fetch('/config');
        config = await response.json();
    } catch (error) {
        console.error('Failed to load configuration:', error);
        throw error;
    }
}

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
        // Ensure config is loaded
        if (!config) {
            await initConfig();
        }
        
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
    peerConnection = new RTCPeerConnection(RTCConfiguration);

    // Add local stream to peer connection
    localStream.getTracks().forEach(track => {
        console.log('Adding track:', track);
        peerConnection.addTrack(track, localStream);
    });

    console.log('ICE gathering state:', peerConnection.iceGatheringState);
    console.log('ICE connection state:', peerConnection.iceConnectionState);

    // Handle remote stream
    peerConnection.ontrack = event => {
        console.log('Received remote stream:', event.streams[0]);
        remoteVideo.srcObject = event.streams[0];
    };

    // // Handle connection state changes
    // peerConnection.onconnectionstatechange = () => {
    //     console.log('Connection state:', peerConnection.connectionState);
        
        
    // };

    // peerConnection.oniceconnectionstatechange = () => {
    //     console.log('ICE connection state:', peerConnection.iceConnectionState);
    // };

    peerConnection.onicecandidate = async (event) => {
        console.log('ICE candidate:', event.candidate);
        console.log("sending string: ", JSON.stringify(event.candidate));
        if (event.candidate) {
            const iceCandidate = {
                candidate: event.candidate.candidate,
                sdpMid: event.candidate.sdpMid,
                sdpMLineIndex: event.candidate.sdpMLineIndex,
                usernameFragment: event.candidate.usernameFragment
            };
            
            await fetch(`${config.videoProcessorUrl}/ice_candidate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(iceCandidate)
            });
        }
    };

    try {
        // const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription();
        // await new Promise((resolve) => {
        //     if (peerConnection.iceGatheringState === 'complete') {
        //         resolve();
        //     } else {
        //         function checkState() {
        //             if (peerConnection.iceGatheringState === 'complete') {
        //                 console.log('ICE gathering state:', peerConnection.iceGatheringState);
        //                 peerConnection.removeEventListener('icegatheringstatechange', checkState);
        //                 resolve();
        //             }
        //         }
        //         peerConnection.addEventListener('icegatheringstatechange', checkState);
        //     }
        // });
        var offer_1 = peerConnection.localDescription;
        const response = await fetch(`${config.videoProcessorUrl}/offer?` + new URLSearchParams({
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
    // peerConnection = null;
    hangupButton.disabled = true;
    callButton.disabled = false;
    remoteVideo.srcObject = null;
}

// Event listeners
startButton.addEventListener('click', start);
callButton.addEventListener('click', startProcessing);
hangupButton.addEventListener('click', hangup); 