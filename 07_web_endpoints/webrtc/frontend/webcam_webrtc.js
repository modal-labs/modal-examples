let ws;
let localStream;
let peerConnection;
let iceServers;
let peerID; // reset per connection
let iceServerType = 'stun';

// Add status display element
const statusDisplay = document.getElementById('statusDisplay');
const MAX_STATUS_HISTORY = 100;
let statusHistory = [];

// Function to update status
function updateStatus(message) {
    
    console.log(message);

    // Add timestamp to message
    const now = new Date();
    const timestamp = now.toLocaleTimeString();
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
        urls: [
            "stun:stun.l.google.com:19302",
          ],
    },
]

// DOM elements
const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');
const startWebcamButton = document.getElementById('startWebcamButton');
const startStreamingButton = document.getElementById('startStreamingButton');
const stopStreamingButton = document.getElementById('stopStreamingButton');

startWebcamButton.disabled = false;
startStreamingButton.disabled = true;
stopStreamingButton.disabled = true;

// Add event listener for STUN/TURN server radio buttons
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

    startWebcamButton.disabled = true;
    startStreamingButton.disabled = true;
    stopStreamingButton.disabled = false;

    peerID = crypto.randomUUID().slice(0, 4);

    updateStatus('Loading YOLO GPU inference in the cloud (this can take up to 20 seconds)...');

    negotiate();

}    

async function negotiate() {
    try {
        // setup websocket connection
        ws_connected = false;
        ws = new WebSocket(`/ws/${peerID}`);

        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
            ws.close();
        };
    
        ws.onclose = function() {
            console.log('WebSocket connection closed');
            if (ws.readyState === WebSocket.OPEN) {
                console.log('Closing websocket connection...');
                ws.close();
            }
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            
            if (msg.type === 'answer') {
                updateStatus('Establishing WebRTC connection...');
                peerConnection.setRemoteDescription(msg);
            } else if (msg.type === 'turn_servers') {
                iceServers = msg.ice_servers;
            } else {
                console.error('Unexpected response from server:', msg);
            }
        };

        console.log('Waiting for websocket to open...');
        await new Promise((resolve) => {
            if (ws.readyState === WebSocket.OPEN) {
                resolve();
            } else {
                ws.addEventListener('open', () => resolve(), { once: true });
            }
        });

        if (iceServerType === 'turn') {
            ws.send(JSON.stringify({type: 'get_turn_servers', peer_id: peerID}));
        } else {
            iceServers = iceSTUNservers;
        }
        // Wait until we have ICE servers
        if (iceServerType === 'turn') {
            await new Promise((resolve) => {
                const checkIceServers = () => {
                    if (iceServers) {
                        resolve();
                    } else {
                        setTimeout(checkIceServers, 100);
                    }
                };
                checkIceServers();
            });
        }

        const rtcConfiguration = {
            iceServers: iceServers,
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

        // Handle ICE candidates using Trickle ICE pattern
        peerConnection.onicecandidate = async (event) => {
            if (!event.candidate || !event.candidate.candidate) {
                return;
            }
            
            const iceCandidate = {
                peer_id: peerID,
                candidate_sdp: event.candidate.candidate, // sdp string representation of candidate
                sdpMid: event.candidate.sdpMid,
                sdpMLineIndex: event.candidate.sdpMLineIndex,
                usernameFragment: event.candidate.usernameFragment
            };

            console.log('Sending ICE candidate: ', iceCandidate.candidate_sdp);
            
            // send ice candidate over ws
            ws.send(JSON.stringify({type: 'ice_candidate', candidate: iceCandidate}));
        };

        peerConnection.onconnectionstatechange = async () => {
            updateStatus(`WebRTCConnection state: ${peerConnection.connectionState}`);
            if (peerConnection.connectionState === 'connected') {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.close();
                }
            }
        };

        // set local description and send as offer to peer
        console.log('Setting local description...');
        await peerConnection.setLocalDescription();
        var offer = peerConnection.localDescription;
        
        console.log('Sending offer...');
        // send offer over ws
        ws.send(JSON.stringify({peer_id: peerID, type: 'offer', sdp: offer.sdp}));

    } catch (e) {
        console.error('Error negotiating:', e);
        alert(e);
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.close();
        }
    }
}

// Stop streaming
async function stop_streaming() {
    await cleanup();
    stopStreamingButton.disabled = true;
    startStreamingButton.disabled = false;
    remoteVideo.srcObject = null;
    updateStatus('Streaming stopped.');
}

// cleanup
async function cleanup() {
    console.log('Cleaning up...');
    iceServers = null;
    if (peerConnection) {
        console.log('Peer Connection state:', peerConnection.connectionState);
        await peerConnection.close();
        peerConnection = null;
    }
    if (ws.readyState === WebSocket.OPEN) {
        await ws.close();
        ws = null;
    }
    
}

// Event listeners
startWebcamButton.addEventListener('click', startWebcam);
startStreamingButton.addEventListener('click', startStreaming);
stopStreamingButton.addEventListener('click', stop_streaming); 

// Add cleanup handler for when browser tab is closed
window.addEventListener('beforeunload', async () => {
    await cleanup();
    // ensure stun/turn radio and iceServerType are reset
    document.querySelectorAll('input[name="iceServer"]').forEach(radio => {
        if (radio.value == "turn") {
            radio.checked = false;
        } else {
            radio.checked = true;
        }
    });
    iceServerType = 'stun';
});