# ---
# lambda-test: false  # auxiliary-file
# ---

import asyncio
import json
import queue
from abc import ABC, abstractmethod
from typing import Optional

import modal
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketState


class ModalWebRtcPeer(ABC):
    """
    Base class for implementing WebRTC peer connections in Modal using aiortc.
    Implement using the `app.cls` decorator.

    This class provides a complete WebRTC peer implementation that handles:
    - Peer connection lifecycle management (creation, negotiation, cleanup)
    - Signaling via Modal Queue for SDP offer/answer exchange and ICE candidate handling
    - Automatic STUN server configuration (defaults to Google's STUN server)
    - Stream setup and management

    Required methods to override:
    - setup_streams(): Implementation for setting up media tracks and streams

    Optional methods to override:
    - initialize(): Custom initialization logic when peer is created
    - run_streams(): Implementation for stream runtime logic
    - get_turn_servers(): Implementation to provide custom TURN server configuration
    - exit(): Custom cleanup logic when peer is shutting down

    The peer connection is established through a ModalWebRtcSignalingServer that manages
    the signaling process between this peer and client peers.
    """

    @modal.enter()
    async def _initialize(self):
        import shortuuid

        self.id = shortuuid.uuid()
        self.pcs = {}
        self.pending_candidates = {}

        # call custom init logic
        await self.initialize()

    async def initialize(self):
        """Override to add custom logic when creating a peer"""

    @abstractmethod
    async def setup_streams(self, peer_id):
        """Override to add custom logic when creating a connection and setting up streams"""
        raise NotImplementedError

    async def run_streams(self, peer_id):
        """Override to add custom logic when running streams"""

    async def get_turn_servers(self, peer_id=None, msg=None) -> Optional[list]:
        """Override to customize TURN servers"""

    async def _setup_peer_connection(self, peer_id):
        """Creates an RTC peer connection via an ICE server"""
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection

        # aiortc automatically uses google's STUN server,
        # but we can also specify our own
        config = RTCConfiguration(
            iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")]
        )
        self.pcs[peer_id] = RTCPeerConnection(configuration=config)
        self.pending_candidates[peer_id] = []
        await self.setup_streams(peer_id)

        print(
            f"{self.id}: Created peer connection and setup streams from {self.id} to {peer_id}"
        )

    @modal.method()
    async def run(self, q: modal.Queue, peer_id: str):
        """Run the RTC peer after establishing a connection by passing WebSocket messages over a Queue."""
        print(f"{self.id}: Running modal peer instance for client peer: {peer_id}...")

        await self._connect_over_queue(q, peer_id)
        await self._run_streams(peer_id)

    async def _connect_over_queue(self, q, peer_id):
        """Connect this peer to another by passing messages along a Modal Queue."""

        msg_handlers = {  # message types we need to handle
            "offer": self.handle_offer,  # SDP offer
            "ice_candidate": self.handle_ice_candidate,  # trickled ICE candidate
            "identify": self.get_identity,  # identify challenge
            "get_turn_servers": self.get_turn_servers,  # TURN server request
        }

        while True:
            try:
                if self.pcs.get(peer_id) and (
                    self.pcs[peer_id].connectionState
                    in ["connected", "closed", "failed"]
                ):
                    print(f"{self.id}: Closing connection to {peer_id} over queue...")
                    q.put("close", partition="server")
                    break

                # read and parse websocket message passed over queue
                msg = json.loads(await q.get.aio(partition=peer_id, timeout=0.5))
                # dispatch the message to its handler
                if handler := msg_handlers.get(msg.get("type")):
                    response = await handler(peer_id, msg)
                else:
                    print(f"{self.id}: Unknown message type: {msg.get('type')}")
                    response = None

                # pass the message back over the queue to the server
                if response is not None:
                    await q.put.aio(json.dumps(response), partition="server")
            except queue.Empty:
                print(f"{self.id}: Queue empty, waiting for message...")
                pass
            except Exception as e:
                print(
                    f"{self.id}: Error handling message from {peer_id}: {type(e)}: {e}"
                )
                continue

    async def _run_streams(self, peer_id):
        """Run WebRTC streaming with a peer."""
        print(f"{self.id}:  running streams to {peer_id}...")

        await self.run_streams(peer_id)

        # run until connection is closed or broken
        while self.pcs[peer_id].connectionState == "connected":
            await asyncio.sleep(0.1)

        print(f"{self.id}:  ending streaming to {peer_id}")

    async def handle_offer(self, peer_id, msg):
        """Handles a peers SDP offer message by producing an SDP answer."""
        from aiortc import RTCSessionDescription

        print(f"{self.id}:  handling SDP offer from {peer_id}...")

        await self._setup_peer_connection(peer_id)
        await self.pcs[peer_id].setRemoteDescription(
            RTCSessionDescription(msg["sdp"], msg["type"])
        )
        answer = await self.pcs[peer_id].createAnswer()
        await self.pcs[peer_id].setLocalDescription(answer)
        sdp = self.pcs[peer_id].localDescription.sdp

        return {"sdp": sdp, "type": answer.type, "peer_id": self.id}

    async def handle_ice_candidate(self, peer_id, msg):
        """Add an ICE candidate sent by a peer."""
        from aiortc import RTCIceCandidate
        from aiortc.sdp import candidate_from_sdp

        candidate = msg.get("candidate")

        if not candidate:
            raise ValueError

        print(
            f"{self.id}:  received ice candidate from {peer_id}: {candidate['candidate_sdp']}..."
        )

        # parse ice candidate
        ice_candidate: RTCIceCandidate = candidate_from_sdp(candidate["candidate_sdp"])
        ice_candidate.sdpMid = candidate["sdpMid"]
        ice_candidate.sdpMLineIndex = candidate["sdpMLineIndex"]

        if not self.pcs.get(peer_id):
            self.pending_candidates[peer_id].append(ice_candidate)
        else:
            if len(self.pending_candidates[peer_id]) > 0:
                [
                    await self.pcs[peer_id].addIceCandidate(c)
                    for c in self.pending_candidates[peer_id]
                ]
                self.pending_candidates[peer_id] = []
            await self.pcs[peer_id].addIceCandidate(ice_candidate)

    async def get_identity(self, peer_id=None, msg=None):
        """Reply to an identify message with own id."""
        return {"type": "identify", "peer_id": self.id}

    async def generate_offer(self, peer_id):
        print(f"{self.id}:  generating offer for {peer_id}...")

        await self._setup_peer_connection(peer_id)
        offer = await self.pcs[peer_id].createOffer()
        await self.pcs[peer_id].setLocalDescription(offer)
        sdp = self.pcs[peer_id].localDescription.sdp

        return {"sdp": sdp, "type": offer.type, "peer_id": self.id}

    async def handle_answer(self, peer_id, answer):
        from aiortc import RTCSessionDescription

        print(f"{self.id}:  handling answer from {peer_id}...")
        # set remote peer description
        await self.pcs[peer_id].setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )

    @modal.exit()
    async def _exit(self):
        print(f"{self.id}: Shutting down...")
        await self.exit()

        if self.pcs:
            print(f"{self.id}: Closing peer connections...")
            await asyncio.gather(*[pc.close() for pc in self.pcs.values()])
            self.pcs = {}

    async def exit(self):
        """Override with any custom logic when shutting down container."""


class ModalWebRtcSignalingServer:
    """
    WebRTC signaling server implementation that mediates connections between client peers
    and Modal-based WebRTC peers. Implement using the `app.cls` decorator.

    This server:
    - Provides a WebSocket endpoint (/ws/{peer_id}) for client connections
    - Spawns Modal-based peer instances for each client connection
    - Handles the WebRTC signaling process by relaying messages between clients and Modal peers
    - Manages the lifecycle of Modal peer instances

    To use this class:
    1. Create a subclass implementing get_modal_peer_class() to return your ModalWebRtcPeer implementation
    2. Optionally override initialize() for custom server setup
    3. Optionally add a frontend route to the `web_app` attribute
    """

    @modal.enter()
    def _initialize(self):
        self.web_app = FastAPI()

        # handle signaling through websocket endpoint
        @self.web_app.websocket("/ws/{peer_id}")
        async def ws(client_websocket: WebSocket, peer_id: str):
            try:
                await client_websocket.accept()
                print(f"Server: Accepted websocket connection from {peer_id}...")
                await self._mediate_negotiation(client_websocket, peer_id)
            except Exception as e:
                print(
                    f"Server: Error accepting websocket connection from {peer_id}: {type(e)}: {e}"
                )
                await client_websocket.close()

        self.initialize()

    def initialize(self):
        pass

    @abstractmethod
    def get_modal_peer_class(self) -> type[ModalWebRtcPeer]:
        """
        Abstract method to return the `ModalWebRtcPeer` implementation to use.
        """
        raise NotImplementedError(
            "Implement `get_modal_peer` to use `ModalWebRtcSignalingServer`"
        )

    @modal.asgi_app()
    def web(self):
        return self.web_app

    async def _mediate_negotiation(self, websocket: WebSocket, peer_id: str):
        modal_peer_class = self.get_modal_peer_class()
        if not any(
            base.__name__ == "ModalWebRtcPeer" for base in modal_peer_class.__bases__
        ):
            raise ValueError(
                "Modal peer class must be an implementation of `ModalWebRtcPeer`"
            )

        with modal.Queue.ephemeral() as q:
            print(f"Server: Spawning modal peer instance for client peer {peer_id}...")
            modal_peer = modal_peer_class()
            modal_peer.run.spawn(q, peer_id)

            await asyncio.gather(
                relay_websocket_to_queue(websocket, q, peer_id),
                relay_queue_to_websocket(websocket, q, peer_id),
            )


async def relay_websocket_to_queue(websocket: WebSocket, q: modal.Queue, peer_id: str):
    while True:
        try:
            # get websocket message off queue and parse as json
            msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
            await q.put.aio(msg, partition=peer_id)
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            if WebSocketState.DISCONNECTED in [
                websocket.application_state,
                websocket.client_state,
            ]:
                print("Server: Websocket connection closed")
                return
            else:
                print(f"Server: Error relaying from websocket to queue: {type(e)}: {e}")


async def relay_queue_to_websocket(websocket: WebSocket, q: modal.Queue, peer_id: str):
    while True:
        try:
            # get websocket message off queue and parse from json
            modal_peer_msg = await q.get.aio(partition="server", timeout=0.5)
            if modal_peer_msg.startswith("close"):
                print(
                    "Server: Close received on queue, closing websocket connection..."
                )
                await websocket.close()
                return

            await websocket.send_text(modal_peer_msg)
        except queue.Empty:
            pass
        except Exception as e:
            if WebSocketState.DISCONNECTED in [
                websocket.application_state,
                websocket.client_state,
            ]:
                print("Server: Websocket connection closed")
                return
            else:
                print(f"Server: Error relaying from queue to websocket: {type(e)}: {e}")
