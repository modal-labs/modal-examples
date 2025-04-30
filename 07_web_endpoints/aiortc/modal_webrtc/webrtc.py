from dataclasses import dataclass

import modal

@dataclass
class IceCandidate:
    peer_id: str
    candidate_sdp: str
    sdpMid: str
    sdpMLineIndex: int
    usernameFragment: str

class ModalWebRTCServer:
    """
    Class that handles WebRTC signaling
    between WebRTC clients and a modal app
    that implements the WebRTCPeer class
    """
    modal_peer_app_name: str = modal.parameter() # not using this, but feels like i should?
    modal_peer_cls_name: str = modal.parameter()

    @modal.enter()
    def initialize(self):

        import uuid

        from fastapi import FastAPI, WebSocket

        self.modal_peer_id = str(uuid.uuid4()) # no longer used but seems worth keeping around
        self.web_app = FastAPI()
        
        # handling signaling through websocket
        @self.web_app.websocket("/ws/{peer_id}")
        async def ws_negotiation(client_websocket: WebSocket, peer_id: str):

            # accept websocket connection
            await client_websocket.accept()         

            # mediate negotiation
            await self.mediate_negotiation(
                client_websocket,
                peer_id
            )


    async def mediate_negotiation(self, client_websocket, peer_id: str):
            
            import asyncio

            from fastapi import WebSocketDisconnect

            if not self.modal_peer_app_name or not self.modal_peer_cls_name:
                print("Modal peer class or app name or class name not set")
                return
            
            self.ModalPeerCls = modal.Cls.from_name(self.modal_peer_app_name, self.modal_peer_cls_name)
            modal_peer_instance = self.ModalPeerCls()

            with modal.Queue.ephemeral() as q:

                print(f"Spawning modal peer instance for client peer {peer_id}...")
                modal_peer_instance.run_with_queue.spawn(q, peer_id)
      

                async def relay_client_messages(client_websocket, q):

                    # handle websocket messages and loop for lifetime
                    while True:
                        
                        try:
                            # get websocket message and parse as json
                            msg = await client_websocket.receive_text()
                            await q.put.aio(
                                msg,
                                partition=peer_id
                            )
                        except Exception as e:
                            if isinstance(e, WebSocketDisconnect):
                                break
                            print(f"Error: {e}")

                async def relay_modal_peer_messages(client_websocket, q):

                    # handle websocket messages and loop for lifetime
                    while True:
                        
                        try:
                            # get websocket message and parse as json
                            modal_peer_msg = await q.get.aio(partition='server')

                            await client_websocket.send_text(modal_peer_msg)

                        except Exception as e:
                            if isinstance(e, WebSocketDisconnect):
                                break
                            else:
                                print(f"Error: {e}")
                
                await asyncio.gather(
                    relay_client_messages(client_websocket, q),
                    relay_modal_peer_messages(client_websocket, q)
                )

            await client_websocket.close()

class ModalWebRTCPeer:
    """
    Base class for WebRTC peer connections using aiortc 
    that handles connection setup, negotiation, and stream management.

    This class provides the core WebRTC functionality including:
    - Peer connection initialization and cleanup
    - Signaling endpoints via HTTP and WebSocket
      - SDP offer/answer exchange
      - Trickle ICE candidate handling
    - Stream setup and management
    
    Subclasses can implement the following methods:
    - initialize(): Any custom initialization logic
    - setup_streams(): Logic for setting up media tracks and streams (this is where the main business logic goes)
    - run_streams(): Logic for starting streams (not always necessary)
    - exit(): Any custom cleanup logic
    """

    @modal.enter()
    async def _initialize(self):

        import asyncio
        import uuid
        import json
        
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from aiortc.sdp import candidate_from_sdp

        self.id = str(uuid.uuid4())
        self.web_app = FastAPI()
        self.pcs = {}

        # HTTP NEGOTIATION ENDPOINT
        
        # handle ice candidate (trickle ice)
        @self.web_app.post("/ice_candidate")
        async def ice_candidate(candidate: IceCandidate):

            if not candidate:
                return 
            
            print(f"Peer {self.id} received ice candidate from {candidate.peer_id}")
            
            peer_id = candidate.peer_id
            
            ice_candidate = candidate_from_sdp(candidate.candidate_sdp)
            ice_candidate.sdpMid = candidate.sdpMid
            ice_candidate.sdpMLineIndex = candidate.sdpMLineIndex
            
            await self.handle_ice_candidate(peer_id, ice_candidate)

        @self.web_app.get("/offer")
        async def offer(peer_id: str, sdp: str, type: str):
            
            if type != "offer":
                return {"error": "Invalid offer type"}
            await self.handle_offer(peer_id, {"sdp": sdp, "type": type})
            return self.generate_answer(peer_id)

        # run until finished
        @self.web_app.post("/run_streams")
        async def run_streams(peer_id: str):
            await self._run_streams(peer_id)
        
        # handling signaling through websocket
        @self.web_app.websocket("/ws/{peer_id}")
        async def ws_negotiation(websocket: WebSocket, peer_id: str):

            # accept websocket connection
            await websocket.accept()

            await self.ws_negotiation(websocket, peer_id)

            # run until complete
            await self._run_streams(peer_id)

        # call custom init logic
        await self.initialize()

    @modal.asgi_app()
    def web_endpoints(self):

        return self.web_app

    async def ws_negotiation(self, websocket, peer_id: str):

        import json
        import asyncio

        from fastapi import WebSocketDisconnect
        from aiortc.sdp import candidate_from_sdp
        
        # handle websocket messages and loop for lifetime
        while True:
            
            try:
                # get websocket message and parse as json
                msg = json.loads(await websocket.receive_text())

                # handle offer
                if msg.get("type") == "offer":
                    
                    print(f"Peer {self.id} received offer from {peer_id}...")

                    await self.handle_offer(peer_id, msg)
                        
                    # generate and send answer
                    await websocket.send_text(
                        json.dumps(self.generate_answer(peer_id))
                    )

                # handle ice candidate (trickle ice)
                elif msg.get("type") == "ice_candidate":

                    candidate = msg.get("candidate")
                    
                    if not candidate or not self.pcs.get(peer_id):
                        return 
                    
                    print(f"Peer {self.id} received ice candidate from {peer_id}...")
                    
                    # parse ice candidate
                    ice_candidate = candidate_from_sdp(candidate["candidate_sdp"])
                    ice_candidate.sdpMid = candidate["sdpMid"]
                    ice_candidate.sdpMLineIndex = candidate["sdpMLineIndex"]
                    
                    await self.handle_ice_candidate(peer_id, ice_candidate)

                    # wait and break if connected
                    # this ensures that we close websocket asap (could remove)
                    await asyncio.sleep(0.2) 
                    if self.pcs[peer_id].connectionState == "connected":
                        break
                
                # get peer's id
                elif msg.get("type") == "identify":

                    await websocket.send_text(json.dumps({"type": "identify", "peer_id": self.id}))
                
                else:
                    print(f"Unknown message type: {msg.get('type')}")

            except Exception as e:
                if isinstance(e, WebSocketDisconnect):
                    break
                else:
                    print(f"Error: {e}")
        
        await websocket.close()
        print("Websocket connection closed")

    @modal.method()
    async def run_with_queue(self, q: modal.Queue, peer_id: str):
        
        import json
        import asyncio

        from aiortc.sdp import candidate_from_sdp
        
        print(f"Running modal peer instance for client peer {peer_id}...")
        # handle websocket messages and loop for lifetime
        while True:
            
            try:
                # get websocket message and parse as json
                msg = json.loads(await q.get.aio(partition=peer_id))

                # handle offer
                if msg.get("type") == "offer":
                    
                    print(f"Peer {self.id} received offer from {peer_id}...")

                    await self.handle_offer(peer_id, msg)
                        
                    # generate and send answer
                    await q.put.aio(
                        json.dumps(self.generate_answer(peer_id)),
                        partition='server'
                    )

                # handle ice candidate (trickle ice)
                elif msg.get("type") == "ice_candidate":

                    candidate = msg.get("candidate")
                    
                    if not candidate or not self.pcs.get(peer_id):
                        return 
                    
                    print(f"Peer {self.id} received ice candidate from {peer_id}...")
                    
                    # parse ice candidate
                    ice_candidate = candidate_from_sdp(candidate["candidate_sdp"])
                    ice_candidate.sdpMid = candidate["sdpMid"]
                    ice_candidate.sdpMLineIndex = candidate["sdpMLineIndex"]
                    
                    await self.handle_ice_candidate(peer_id, ice_candidate)

                    # wait and break if connected
                    # this ensures that we close websocket asap (could remove)
                    await asyncio.sleep(0.2) 
                    if self.pcs[peer_id].connectionState == "connected":
                        break
                
                # get peer's id
                elif msg.get("type") == "identify":

                    await q.put.aio(
                        json.dumps({"type": "identify", "peer_id": self.id}),
                        partition='server'
                    )
                
                else:
                    print(f"Unknown message type: {msg.get('type')}")

            except Exception as e:
                print(f"Error: {e}")
        
        while self.pcs[peer_id].connectionState == "connected":
            await asyncio.sleep(1.0)

        print(f"Shutting down modal peer instance for client peer {peer_id}...")

    async def initialize(self):
        """
        Any custom logic when instantiating the peer
        """
        pass

    async def _setup_peer_connection(self, peer_id):

        from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer

        # create peer connection with STUN server
        # aiortc automatically uses google's STUN server when
        # self.pcs[peer_id] = RTCPeerConnection()
        # is called, but we can also specify our own:
        config = RTCConfiguration([RTCIceServer(urls="stun:stun.l.google.com:19302")])
        self.pcs[peer_id] = RTCPeerConnection(configuration = config)

        await self.setup_streams(peer_id)

        print(f"Created peer connection and setup streams from {self.id} to {peer_id}")

    async def setup_streams(self, peer_id):
        """
        Any custom logic when setting up the connection and streams
        """
        pass

    async def generate_offer(self, peer_id):

        print(f"Peer {self.id} generating offer for {peer_id}...")

        # initalize peer connection
        await self._setup_peer_connection(peer_id)
        # create initial offer
        offer = await self.pcs[peer_id].createOffer()
        # set local/our description, this also triggers and waits for ICE gathering/generation of ICE candidate info
        await self.pcs[peer_id].setLocalDescription(offer)
        # NOTE: we can't use `offer.sdp` because the ICE candidates are not included
        # these are embedded in the SDP after setLocalDescription() is called
        return {"sdp": self.pcs[peer_id].localDescription.sdp, "type": offer.type, "peer_id": self.id}

    async def handle_offer(self, peer_id, offer):

        from aiortc import RTCSessionDescription

        print(f"Peer {self.id} handling offer from {peer_id}...")

        # initalize peer connection and streams
        await self._setup_peer_connection(peer_id)
        # set remote description
        await self.pcs[peer_id].setRemoteDescription(RTCSessionDescription(offer["sdp"], offer["type"]))
        # create answer
        answer = await self.pcs[peer_id].createAnswer()
        # set local/our description, this also triggers ICE gathering
        await self.pcs[peer_id].setLocalDescription(answer)

    def generate_answer(self, peer_id):

        print(f"Peer {self.id} generating answer for {peer_id}...")

        return {
            "sdp": self.pcs[peer_id].localDescription.sdp, 
            "type": "answer", 
            "peer_id": self.id
        }
    
    async def handle_answer(self, peer_id, answer):

        from aiortc import RTCSessionDescription

        print(f"Peer {self.id} handling answer from {peer_id}...")
        # set remote peer description
        await self.pcs[peer_id].setRemoteDescription(
            RTCSessionDescription(
                sdp = answer["sdp"], 
                type = answer["type"]
            )
        )

    async def handle_ice_candidate(self, peer_id, candidate):

        import asyncio

        print(f"Peer {self.id} handling ice candidate from {peer_id}...")

        # sometimes this event is called before 
        # the peer connection is created on this end
        retries = 5
        while not self.pcs.get(peer_id) and retries > 0:
            await asyncio.sleep(0.1)
            retries -= 1

        if not retries:
            print(f"Peer {self.id} failed to create peer connection for {peer_id} before ICE candidate event")
            return

        await self.pcs[peer_id].addIceCandidate(candidate)

    async def _run_streams(self, peer_id):

        import asyncio 

        print(f"Peer {self.id} running streams for {peer_id}...")

        # trigger custom streams if necessary
        await self.run_streams(peer_id)

        # run until connection is closed
        while self.pcs[peer_id].connectionState == "connected":
            await asyncio.sleep(1.0)

    # custom logic for running streams
    async def run_streams(self, peer_id):
        pass

    @modal.exit()
    async def _exit(self):

        import asyncio

        print(f"Shutting down peer: {self.id}...")
        # call custom exit logic
        await self.exit()

        # close peer connections
        if self.pcs:
            print(f"Closing peer connections for peer {self.id}...")
            await asyncio.gather(*[pc.close() for pc in self.pcs.values()])
            self.pcs = {}

    async def exit(self):
        """
        Any custom logic when shutting down container
        """
        pass