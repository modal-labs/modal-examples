let ws;
let localStream;
let peerConnection;
let iceServers;
const peerID = crypto.randomUUID();
let iceServerType = 'stun';

// Add status display element
const statusDisplay = document.getElementById('statusDisplay');
const MAX_STATUS_HISTORY = 10;
let statusHistory = [];

// Function to update status
function updateStatus(message) {

    // Add timestamp to message
    const now = new Date();
    const timestamp = now.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: 'numeric', hour12: true });
    const statusLine = `[${timestamp}] ${message}`;
    
    // Add to history
    statusHistory.push(statusLine);
    
    // Keep only last 10 messages
    if (statusHistory.length > MAX_STATUS_HISTORY) {
        statusHistory.shift();
    }
    
    // Update display
    statusDisplay.innerHTML = statusHistory.map(line => 
        `<div class="status-line">${line}</div>`
    ).join('');
    
    // Scroll to bottom
    statusDisplay.scrollTop = statusDisplay.scrollHeight;
}

const iceSTUNservers = [
    {
        urls: "stun:stun.l.google.com:19302",
    },
]

// DOM elements
const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');
const startWebcamButton = document.getElementById('startWebcamButton');
const startStreamingButton = document.getElementById('startStreamingButton');
const stopStreamingButton = document.getElementById('stopStreamingButton');

// Add event listener for ICE server radio buttons
document.querySelectorAll('input[name="iceServer"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        iceServerType = e.target.value;
        console.log('ICE server type changed to:', iceServerType);
    });
});

// Get webcam media stream
async function startWebcam() {
    try {
        
        localStream = await navigator.mediaDevices.getUserMedia({ 
            video: {
                // prefer environment facing camera
                facingMode: { ideal: "environment" }
            }, 
            audio: false
        });
        localVideo.srcObject = localStream;
        startWebcamButton.disabled = true;
        startStreamingButton.disabled = false;
    } catch (err) {
        console.error('Error accessing media devices:', err);
    }
}

// Create and set up peer connection
async function startStreaming() {

    updateStatus('[10:30:45] Loading YOLO GPU inference in the cloud...');

    startWebcamButton.disabled = true;
    startStreamingButton.disabled = true;
    stopStreamingButton.disabled = false;

    try {
        negotiate();
    } catch (err) {
        console.error('Error creating offer:', err);
    }
}    

async function negotiate() {
    try {
        // setup websocket connection
        ws = new WebSocket(`/ws/${peerID}`);
        
        console.log('Waiting for websocket to open...');
        await new Promise((resolve) => {
            ws.onopen = resolve;
        });

        console.log('Websocket opened');

        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
            ws.close();
        };
    
        ws.onclose = function() {
            console.log('WebSocket connection closed');
            ws = null;
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            console.log('Received answer:', msg);
            
            if (msg.type === 'answer') {
                if (iceServerType == 'stun'){
                    updateStatus('[10:30:46] Establishing WebRTC connection...');
                }
                peerConnection.setRemoteDescription(msg);
            } else if (msg.type === 'turn_servers') {
                if (iceServerType == 'turn'){
                    updateStatus('[10:30:46] Establishing WebRTC connection...');
                }
                iceServers = msg.ice_servers;
            } else if (msg.type === 'connection_status') {
                if (msg.status === 'connected') {
                    console.log('Connection status:', msg.status);
                    ws.close();
                }
            } else {
                console.error('Unexpected response from server:', msg);
            }
        };

        if (iceServerType === 'turn') {
            
            console.log('Getting TURN servers...');
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'get_turn_servers', peer_id: peerID}));
            }
        } else {
            console.log('Using STUN servers...');
            iceServers = iceSTUNservers;
        }

        // Wait until we have ICE servers
        if (iceServerType === 'turn') {
            await new Promise((resolve) => {
                const checkIceServers = () => {
                    if (iceServers) {
                        console.log('ICE servers received:', iceServers);
                        resolve();
                    } else {
                        setTimeout(checkIceServers, 100);
                    }
                };
                checkIceServers();
            });
        }

        if (!iceServers || !iceServers.length) {
            console.error('No ICE servers received');
            throw new Error('No ICE servers available');
        }
        const rtcConfiguration = {
            iceServers: iceServers
        }
        peerConnection = new RTCPeerConnection(rtcConfiguration);

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
            
            if (event.candidate) {
                const iceCandidate = {
                    peer_id: peerID,
                    candidate_sdp: event.candidate.candidate, // sdp string representation of candidate
                    sdpMid: event.candidate.sdpMid,
                    sdpMLineIndex: event.candidate.sdpMLineIndex,
                    usernameFragment: event.candidate.usernameFragment
                };

                console.log('Sending ICE candidate:', iceCandidate);
                
                // send ice candidate over ws
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'ice_candidate', candidate: iceCandidate}));
                }
            }
        };

        peerConnection.onconnectionstatechange = async () => {
            updateStatus(`[10:30:47] WebRTCConnection state: ${peerConnection.connectionState}`);
            if (peerConnection.connectionState === 'connected') {
                console.log('Connection state:', peerConnection.connectionState);
            }
        };

        // set local description and send as offer to peer
        console.log('Setting local description...');
        await peerConnection.setLocalDescription();
        var offer = peerConnection.localDescription;
        
        console.log('Sending offer...');
        // send offer over ws
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({peer_id: peerID, type: 'offer', sdp: offer.sdp}));
        }

    } catch (e) {
        alert(e);
    }
}

// cleanup
function cleanup() {
    iceServers = null;
    if (peerConnection) {
        peerConnection.close();
        peerConnection = null;
    }
    if (ws) {
        ws.close();
        ws = null;
    }
}

// Stop streaming
function stop_streaming() {
    cleanup();
    stopStreamingButton.disabled = true;
    startStreamingButton.disabled = false;
    remoteVideo.srcObject = null;
}

// Event listeners
startWebcamButton.addEventListener('click', startWebcam);
startStreamingButton.addEventListener('click', startStreaming);
stopStreamingButton.addEventListener('click', stop_streaming); 

// Add cleanup handler for when browser tab is closed
window.addEventListener('beforeunload', () => {
    cleanup();
});