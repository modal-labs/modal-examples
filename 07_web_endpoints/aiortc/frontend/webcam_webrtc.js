//API Key for the credential: 84100256a51e692dea36d11e1b366fac9e00

// Calling the REST API TO fetch the TURN Server Credentials
// function getIceServers() {
//   return fetch("https://modal-test-turn.metered.live/api/v1/turn/credentials?apiKey=84100256a51e692dea36d11e1b366fac9e00")
//     .then(response => response.json());
// }

// let iceServers;
// getIceServers().then(servers => {
//   iceServers = servers;
// });

// need to setup with .env, these are non-sensitive credentials
// const iceTURNServers = [
//         {
//           urls: "stun:stun.relay.metered.ca:80",
//         },
//         {
//           urls: "turn:standard.relay.metered.ca:80",
//           username: "9fe1dc70b0e8f69039113e3b",
//           credential: "v8hbPkad1WKL3Bxj",
//         },
//         {
//           urls: "turn:standard.relay.metered.ca:80?transport=tcp",
//           username: "9fe1dc70b0e8f69039113e3b",
//           credential: "v8hbPkad1WKL3Bxj",
//         },
//         {
//           urls: "turn:standard.relay.metered.ca:443",
//           username: "9fe1dc70b0e8f69039113e3b",
//           credential: "v8hbPkad1WKL3Bxj",
//         },
//         {
//           urls: "turns:standard.relay.metered.ca:443?transport=tcp",
//           username: "9fe1dc70b0e8f69039113e3b",
//           credential: "v8hbPkad1WKL3Bxj",
//         },
//     ]

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

// Implementation type (http or websocket)
let implementationType = 'http';
let iceServerType = 'stun';

// Add event listener for radio buttons
document.querySelectorAll('input[name="implementation"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        implementationType = e.target.value;
        console.log('Implementation type changed to:', implementationType);
    });
});

// Add event listener for ICE server radio buttons
document.querySelectorAll('input[name="iceServer"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        iceServerType = e.target.value;
        console.log('ICE server type changed to:', iceServerType);
    });
});

// WebRTC variables
let ws;
let localStream;
let peerConnection;
let iceServers;
const peerID = crypto.randomUUID();

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


    startWebcamButton.disabled = true;
    startStreamingButton.disabled = true;
    stopStreamingButton.disabled = false;

    try {
        negotiate();
    } catch (err) {
        console.error('Error creating offer:', err);
    }
}    

// async function setupPeerConnection(peerConnection) {}

//     // Add local stream to peer connection
//     localStream.getTracks().forEach(track => {
//         console.log('Adding track:', track);
//         peerConnection.addTrack(track, localStream);
//     });

//     // Handle remote stream when triggered
//     peerConnection.ontrack = event => {
//         console.log('Received remote stream:', event.streams[0]);
//         remoteVideo.srcObject = event.streams[0];
//     };

//     // Handle ICE candidates using ICE trickle pattern
//     // for some devices/networks, waiting for automatic ICE candidate gathering
//     // to complete can take a long time
//     // so we use the ICE trickle pattern to send candidates as they are gathered
//     // which is much, much faster
//     peerConnection.onicecandidate = async (event) => {
//         if (!event.candidate) {
//             return;
//         }
        
//         if (event.candidate) {
//             const iceCandidate = {
//                 peer_id: peerID,
//                 candidate_sdp: event.candidate.candidate, // sdp string representation of candidate
//                 sdpMid: event.candidate.sdpMid,
//                 sdpMLineIndex: event.candidate.sdpMLineIndex,
//                 usernameFragment: event.candidate.usernameFragment
//             };

//             console.log('Sending ICE candidate:', iceCandidate);
            
//             if (implementationType === 'http') {
//                 await fetch(`/ice_candidate`, {
//                     method: 'POST',
//                     headers: {
//                     'Content-Type': 'application/json'
//                     },
//                     body: JSON.stringify(iceCandidate)
//                 });
//             } else {
//                 // send ice candidate over ws
//                 if (ws.readyState === WebSocket.OPEN) {
//                     ws.send(JSON.stringify({type: 'ice_candidate', candidate: iceCandidate}));
//                 }
//             }
//         }
//     };

//     peerConnection.onconnectionstatechange = async () => {

//         if (peerConnection.connectionState === 'connected') {

//             console.log('Connection state:', peerConnection.connectionState);

//             // this is the functional call that runs while streams are processing
//             // when using http signaling
//             if (implementationType === 'http') {
//                 await fetch(`/run_streams?peer_id=${peerID}`, {
//                     method: 'POST',
//                     headers: {
//                         'Content-Type': 'application/json'
//                     }
//                 });
//             }
//         }
//     };

// }

// could use websockets to do this communication
// directly and asynchrounsly - including ice candidate transfer
async function negotiate() {
    

    try {

        if (implementationType === 'websocket') {

            // setup websocket connection
            ws = new WebSocket(`/ws/${peerID}`);

            // wait for websocket to open
            await new Promise((resolve) => {
                ws.onopen = resolve;
            });

            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                ws.close();
            };
        
            ws.onclose = function() {
                console.log('WebSocket connection closed');
                ws.close();
            };

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'answer') {
                    console.log('Received answer:', msg.sdp);
                    peerConnection.setRemoteDescription(msg);
                } else if (msg.type === 'turn_servers') {
                    iceServers = msg.ice_servers;
                } else {
                    console.error('Unexpected response from server:', msg);
                }
            };
        }

        if (iceServerType === 'turn') {
            console.log('Getting TURN servers...');

            let msg;

            if (implementationType === 'http') {
                const iceServersResponse = await fetch(`/turn_servers`, {
                    method: 'GET'
                });
                msg = await iceServersResponse.json();
            } else {
                await ws.send(JSON.stringify({type: 'get_turn_servers', peer_id: peerID}));
            }
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
                
                if (implementationType === 'http') {
                    await fetch(`/ice_candidate`, {
                        method: 'POST',
                        headers: {
                        'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(iceCandidate)
                    });
                } else {
                    // send ice candidate over ws
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({type: 'ice_candidate', candidate: iceCandidate}));
                    }
                }
            }
        };

        peerConnection.onconnectionstatechange = async () => {

            if (peerConnection.connectionState === 'connected') {

                console.log('Connection state:', peerConnection.connectionState);

                // this is the functional call that runs while streams are processing
                // when using http signaling
                if (implementationType === 'http') {
                    await fetch(`/run_streams?peer_id=${peerID}`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    });
                }
            }
        };

        // set local description and send as offer to peer
        console.log('Setting local description...');
        await peerConnection.setLocalDescription();
        var offer = peerConnection.localDescription;
        
        console.log('Sending offer...');
        if (implementationType === 'http') {
            const response = await fetch(`/offer?` + new URLSearchParams({
                peer_id: peerID,
                sdp: offer.sdp,
                type: offer.type
            }), {
                    method: 'GET'
                });
            const answer = await response.json();
            // set remote description
            console.log('Received answer:', answer);
            await peerConnection.setRemoteDescription(answer);
        } else {
            // send offer over ws
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({peer_id: peerID, type: 'offer', sdp: offer.sdp}));
            }
        }

    } catch (e) {
        alert(e);
    }
}

// Stop streaming
function stop_streaming() {
    if (peerConnection) {
        peerConnection.close();
        peerConnection = null;
    }
    if (ws) {
        ws.close();
    }
    stopStreamingButton.disabled = true;
    startStreamingButton.disabled = false;
    remoteVideo.srcObject = null;
}

// Event listeners
startWebcamButton.addEventListener('click', startWebcam);
startStreamingButton.addEventListener('click', startStreaming);
stopStreamingButton.addEventListener('click', stop_streaming); 