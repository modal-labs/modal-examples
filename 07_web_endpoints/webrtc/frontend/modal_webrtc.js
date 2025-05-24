export class ModalWebRtcClient extends EventTarget {
    constructor() {
        super();
        this.ws = null;
        this.localStream = null;
        this.peerConnection = null;
        this.iceServers = null;
        this.peerID = null;
        this.iceServerType = 'stun';
    }

    updateStatus(message) {
        this.dispatchEvent(new CustomEvent('status', { 
            detail: { message }
        }));
        console.log(message);
    }

    // Get webcam media stream
    async startWebcam() {
        try {
            this.localStream = await navigator.mediaDevices.getUserMedia({ 
                video: {
                    facingMode: { ideal: "environment" }
                }, 
                audio: false
            });
            this.dispatchEvent(new CustomEvent('localStream', { 
                detail: { stream: this.localStream }
            }));
            return this.localStream;
        } catch (err) {
            console.error('Error accessing media devices:', err);
            this.dispatchEvent(new CustomEvent('error', { 
                detail: { error: err }
            }));
            throw err;
        }
    }

    // Create and set up peer connection
    async startStreaming() {
        this.peerID = this.generateShortUUID();
        this.updateStatus('Loading YOLO GPU inference in the cloud (this can take up to 20 seconds)...');
        await this.negotiate();
    }

    async negotiate() {
        try {
            // setup websocket connection
            this.ws = new WebSocket(`/ws/${this.peerID}`);

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.dispatchEvent(new CustomEvent('error', { 
                    detail: { error }
                }));
            };
        
            this.ws.onclose = () => {
                console.log('WebSocket connection closed');
                this.dispatchEvent(new CustomEvent('websocketClosed'));
            };

            this.ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                
                if (msg.type === 'answer') {
                    this.updateStatus('Establishing WebRTC connection...');
                    this.peerConnection.setRemoteDescription(msg);
                } else if (msg.type === 'turn_servers') {
                    this.iceServers = msg.ice_servers;
                } else {
                    console.error('Unexpected response from server:', msg);
                }
            };

            console.log('Waiting for websocket to open...');
            await new Promise((resolve) => {
                if (this.ws.readyState === WebSocket.OPEN) {
                    resolve();
                } else {
                    this.ws.addEventListener('open', () => resolve(), { once: true });
                }
            });

            if (this.iceServerType === 'turn') {
                this.ws.send(JSON.stringify({type: 'get_turn_servers', peer_id: this.peerID}));
            } else {
                this.iceServers = [
                    {
                        urls: ["stun:stun.l.google.com:19302"],
                    },
                ];
            }

            // Wait until we have ICE servers
            if (this.iceServerType === 'turn') {
                await new Promise((resolve) => {
                    const checkIceServers = () => {
                        if (this.iceServers) {
                            resolve();
                        } else {
                            setTimeout(checkIceServers, 100);
                        }
                    };
                    checkIceServers();
                });
            }

            const rtcConfiguration = {
                iceServers: this.iceServers,
            }
            this.peerConnection = new RTCPeerConnection(rtcConfiguration);

            // Add local stream to peer connection
            this.localStream.getTracks().forEach(track => {
                console.log('Adding track:', track);
                this.peerConnection.addTrack(track, this.localStream);
            });

            // Handle remote stream when triggered
            this.peerConnection.ontrack = event => {
                console.log('Received remote stream:', event.streams[0]);
                this.dispatchEvent(new CustomEvent('remoteStream', { 
                    detail: { stream: event.streams[0] }
                }));
            };

            // Handle ICE candidates using Trickle ICE pattern
            this.peerConnection.onicecandidate = async (event) => {
                if (!event.candidate || !event.candidate.candidate) {
                    return;
                }
                
                const iceCandidate = {
                    peer_id: this.peerID,
                    candidate_sdp: event.candidate.candidate,
                    sdpMid: event.candidate.sdpMid,
                    sdpMLineIndex: event.candidate.sdpMLineIndex,
                    usernameFragment: event.candidate.usernameFragment
                };

                console.log('Sending ICE candidate: ', iceCandidate.candidate_sdp);
                
                // send ice candidate over ws
                this.ws.send(JSON.stringify({type: 'ice_candidate', candidate: iceCandidate}));
            };

            this.peerConnection.onconnectionstatechange = async () => {
                const state = this.peerConnection.connectionState;
                this.updateStatus(`WebRTCConnection state: ${state}`);
                this.dispatchEvent(new CustomEvent('connectionStateChange', { 
                    detail: { state }
                }));
                
                if (state === 'connected') {
                    if (this.ws.readyState === WebSocket.OPEN) {
                        this.ws.close();
                    }
                }
            };

            // set local description and send as offer to peer
            console.log('Setting local description...');
            await this.peerConnection.setLocalDescription();
            var offer = this.peerConnection.localDescription;
            
            console.log('Sending offer...');
            // send offer over ws
            this.ws.send(JSON.stringify({peer_id: this.peerID, type: 'offer', sdp: offer.sdp}));

        } catch (e) {
            console.error('Error negotiating:', e);
            this.dispatchEvent(new CustomEvent('error', { 
                detail: { error: e }
            }));
            throw e;
        }
    }

    // Stop streaming
    async stopStreaming() {
        await this.cleanup();
        this.updateStatus('Streaming stopped.');
        this.dispatchEvent(new CustomEvent('streamingStopped'));
    }

    // cleanup
    async cleanup() {
        console.log('Cleaning up...');
        this.iceServers = null;
        if (this.peerConnection) {
            console.log('Peer Connection state:', this.peerConnection.connectionState);
            await this.peerConnection.close();
            this.peerConnection = null;
        }
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            await this.ws.close();
            this.ws = null;
        }
        this.dispatchEvent(new CustomEvent('cleanup'));
    }

    setIceServerType(type) {
        this.iceServerType = type;
        console.log('ICE server type changed to:', this.iceServerType);
        this.dispatchEvent(new CustomEvent('iceServerTypeChanged', { 
            detail: { type }
        }));
    }

    // Generate a short, URL-safe UUID
    generateShortUUID() {
        const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';
        let result = '';
        // Generate 22 characters (similar to short-uuid's default length)
        for (let i = 0; i < 22; i++) {
            result += chars.charAt(Math.floor(Math.random() * chars.length));
        }
        return result;
    }
} 