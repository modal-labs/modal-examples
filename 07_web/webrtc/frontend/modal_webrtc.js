export class ModalWebRtcClient extends EventTarget {
    constructor() {
        super();
        this.localStream = null;
        this.peerConnection = null;
        this.iceServerType = 'stun';
        this.iceGatheringTimeoutMs = 10000;
        this._offerAbort = null;
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
                    width: { ideal: 640 },
                    height: { ideal: 480 },
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
        this.updateStatus('Loading YOLO GPU inference in the cloud (this can take up to 20 seconds)...');
        await this.negotiate();
    }

    async fetchIceServers(iceServerType, signal) {
        const response = await fetch(`/ice-servers?mode=${iceServerType}`, { signal });
        if (!response.ok) {
            throw new Error(`Failed to fetch ICE servers: ${response.status}`);
        }
        const payload = await response.json();
        return payload.ice_servers;
    }

    // Resolve when ICE gathering completes, abort fires, or proceed with partial SDP after timeout.
    waitForIceGathering(pc, signal) {
        if (signal?.aborted || pc.iceGatheringState === 'complete') {
            return Promise.resolve();
        }
        return new Promise((resolve) => {
            const done = () => {
                clearTimeout(timeoutId);
                pc.removeEventListener('icegatheringstatechange', check);
                signal?.removeEventListener('abort', onAbort);
                resolve();
            };
            const timeoutId = setTimeout(() => {
                console.warn(
                    `ICE gathering timed out after ${this.iceGatheringTimeoutMs}ms; continuing with available candidates`
                );
                done();
            }, this.iceGatheringTimeoutMs);

            const check = () => {
                if (pc.iceGatheringState === 'complete') {
                    done();
                }
            };
            const onAbort = () => done();
            pc.addEventListener('icegatheringstatechange', check);
            signal?.addEventListener('abort', onAbort, { once: true });
        });
    }

    async negotiate() {
        // Snapshot ICE mode for this negotiation so a mid-flight radio change
        // cannot desync browser RTCPeerConnection config from the GPU peer.
        const iceServerType = this.iceServerType;
        // Per-negotiation abort/PC so overlapping Stop/Start cannot clear a newer run.
        if (this._offerAbort) {
            this._offerAbort.abort();
        }
        const abort = new AbortController();
        this._offerAbort = abort;
        let pc = null;

        const dropPc = () => {
            if (!pc) return;
            pc.close();
            if (this.peerConnection === pc) {
                this.peerConnection = null;
            }
            pc = null;
        };

        try {
            const iceServers = await this.fetchIceServers(iceServerType, abort.signal);
            if (abort.signal.aborted) {
                return;
            }
            pc = new RTCPeerConnection({ iceServers });
            this.peerConnection = pc;

            // Pipecat peers expect a client datachannel and audio-then-video transceiver order.
            pc.createDataChannel('modal-webrtc');
            pc.addTransceiver('audio');

            // Add local stream to peer connection
            this.localStream.getTracks().forEach(track => {
                console.log('Adding track:', track);
                pc.addTrack(track, this.localStream);
            });

            // Handle remote stream when triggered
            pc.ontrack = event => {
                if (this.peerConnection !== pc) return;
                console.log('Received remote stream:', event.streams[0]);
                this.dispatchEvent(new CustomEvent('remoteStream', { 
                    detail: { stream: event.streams[0] }
                }));
            };

            pc.onconnectionstatechange = () => {
                if (this.peerConnection !== pc) return;
                const state = pc.connectionState;
                this.updateStatus(`WebRTCConnection state: ${state}`);
                this.dispatchEvent(new CustomEvent('connectionStateChange', { 
                    detail: { state }
                }));
            };

            // set local description, gather ICE, then POST offer for an SDP answer
            console.log('Setting local description...');
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            await this.waitForIceGathering(pc, abort.signal);

            if (abort.signal.aborted) {
                dropPc();
                return;
            }

            console.log('Sending offer...');
            const response = await fetch('/offer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                signal: abort.signal,
                body: JSON.stringify({
                    sdp: pc.localDescription.sdp,
                    type: pc.localDescription.type,
                    ice_server_type: iceServerType,
                }),
            });
            if (!response.ok) {
                throw new Error(`Offer failed: ${response.status}`);
            }

            const answer = await response.json();
            if (abort.signal.aborted) {
                dropPc();
                return;
            }
            this.updateStatus('Establishing WebRTC connection...');
            await pc.setRemoteDescription(answer);
        } catch (e) {
            if (e && e.name === 'AbortError') {
                console.log('Offer aborted');
                dropPc();
                return;
            }
            console.error('Error negotiating:', e);
            this.dispatchEvent(new CustomEvent('error', { 
                detail: { error: e }
            }));
            dropPc();
            throw e;
        } finally {
            if (this._offerAbort === abort) {
                this._offerAbort = null;
            }
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
        if (this._offerAbort) {
            this._offerAbort.abort();
            this._offerAbort = null;
        }
        if (this.peerConnection) {
            console.log('Peer Connection state:', this.peerConnection.connectionState);
            this.peerConnection.close();
            this.peerConnection = null;
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
}
