
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

// WebRTC variables
let ws;
let localStream;
let peerConnection;
let iceServers;
const peerID = crypto.randomUUID();
let iceServerType = 'stun';

// Add event listener for ICE server radio buttons
document.querySelectorAll('input[name="iceServer"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        iceServerType = e.target.value;
        console.log('ICE server type changed to:', iceServerType);
    });
});

// Get local media stream
async function startWebcam() {

    try {
        
        localStream = await navigator.mediaDevices.getUserMedia({ 
            video: {
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

    try {
        negotiate();
    } catch (err) {
        console.error('Error creating offer:', err);
    }
}    

// wait for websocket to open
const waitForOpenConnection = (socket) => {

    return new Promise((resolve, reject) => {
        const maxNumberOfAttempts = 10
        const intervalTime = 1000 //ms

        let currentAttempt = 0
        const interval = setInterval(() => {
            if (currentAttempt > maxNumberOfAttempts - 1) {
                clearInterval(interval)
                reject(new Error('Maximum number of attempts exceeded'))
            } else if (socket.readyState === socket.OPEN) {
                clearInterval(interval)
                resolve()
            }
            currentAttempt++
        }, intervalTime)
    })
}

async function negotiate() {
    
    try {
        // setup websocket connection
        ws = new WebSocket(`/ws/${peerID}`);
        
        console.log('Waiting for websocket to open...');
        await waitForOpenConnection(ws);

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
                peerConnection.setRemoteDescription(msg);
            } else if (msg.type === 'turn_servers') {
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