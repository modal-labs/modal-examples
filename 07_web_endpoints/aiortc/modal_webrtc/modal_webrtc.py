from dataclasses import dataclass
from typing import ClassVar

import modal


@dataclass
class IceCandidate:
    peer_id: str
    candidate_sdp: str
    sdpMid: str
    sdpMLineIndex: int
    usernameFragment: str


class ModalWebRtcServer:
    """
    Class that spawns a model peer instance
    that implements the ModalWebRTCPeer class
    and handles WebRTC signaling
    between a WebRTC clients and the modal
    peer instance.
    """

    # need to test if this works via ws endpoint as query params
    modal_peer_app_name: str = modal.parameter(default="")
    modal_peer_cls_name: str = modal.parameter(default="")
    # we use this classvar instead since we have access to it
    modal_peer_cls: ClassVar = None

    @modal.enter()
    def _initialize(self):
        import uuid

        from fastapi import FastAPI, WebSocket

        # no longer used but seems worth keeping around
        self.modal_peer_id = str(uuid.uuid4())
        self.web_app = FastAPI()

        # handling signaling through websocket endpoint
        # can set modal.parameter() values to dymamically
        # choose a ModalWebRTCPeer subclass
        # need to test
        @self.web_app.websocket("/ws/{peer_id}")
        async def ws(client_websocket: WebSocket, peer_id: str):
            print(
                f"Query param, modal_peer_app_name: {client_websocket.query_params.get('modal_peer_app_name')}"
            )
            print(
                f"Query param, modal_peer_cls_name: {client_websocket.query_params.get('modal_peer_cls_name')}"
            )

            self.modal_peer_app_name = client_websocket.query_params.get(
                "modal_peer_app_name"
            )
            self.modal_peer_cls_name = client_websocket.query_params.get(
                "modal_peer_cls_name"
            )
            # accept websocket connection
            await client_websocket.accept()

            # mediate negotiation
            await self._mediate_negotiation(client_websocket, peer_id)

        self.initialize()

    def initialize(self):
        pass

    @modal.asgi_app()
    def web_endpoints(self):
        return self.web_app

    async def _mediate_negotiation(self, client_websocket, peer_id: str):
        import asyncio

        from fastapi import WebSocketDisconnect

        if self.modal_peer_app_name and self.modal_peer_cls_name:
            print(
                f"Looking up deployed class by params: {self.modal_peer_app_name}, {self.modal_peer_cls_name}"
            )
            self.modal_peer_cls = modal.Cls.from_name(
                self.modal_peer_app_name, self.modal_peer_cls_name
            )

        if self.modal_peer_cls:
            modal_peer_instance = self.modal_peer_cls()
        else:
            print("Modal peer class not set.")
            return

        with modal.Queue.ephemeral() as q:
            print(f"Spawning modal peer instance for client peer {peer_id}...")
            modal_peer_instance.run_with_queue.spawn(q, peer_id)

            async def relay_client_messages(client_websocket, q):
                # handle websocket messages and loop for lifetime
                while True:
                    try:
                        # get websocket message and parse as json
                        msg = await asyncio.wait_for(
                            client_websocket.receive_text(), timeout=2
                        )

                        await q.put.aio(msg, partition=peer_id)
                    except Exception as e:
                        if isinstance(e, TimeoutError):
                            continue
                        elif not isinstance(e, WebSocketDisconnect):
                            print(
                                f"Error relaying from client peer to modal peer {peer_id}: {e}"
                            )
                        return

            async def relay_modal_peer_messages(client_websocket, q):
                # handle websocket messages and loop for lifetime
                while True:
                    try:
                        # get websocket message and parse as json
                        modal_peer_msg = await asyncio.wait_for(
                            q.get.aio(partition="server"), timeout=2
                        )

                        if modal_peer_msg.startswith("close"):
                            print(
                                f"Server closing websocket connection to client peer {peer_id}..."
                            )
                            await client_websocket.close()
                            return

                        await client_websocket.send_text(modal_peer_msg)

                    except Exception as e:
                        if isinstance(e, TimeoutError):
                            continue
                        else:
                            print(
                                f"Error relaying from modal peer to client peer {peer_id}: {e}"
                            )
                            return

            await asyncio.gather(
                relay_client_messages(client_websocket, q),
                relay_modal_peer_messages(client_websocket, q),
            )


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
        import uuid

        self.id = str(uuid.uuid4())
        self.pcs = {}

        # call custom init logic
        await self.initialize()

    async def initialize(self):
        """
        Any custom logic when instantiating the peer
        """
        pass

    @modal.method()
    async def run_with_queue(self, q: modal.Queue, peer_id: str):
        import asyncio
        import json

        from aiortc.sdp import candidate_from_sdp

        print(f"Running modal peer instance for client peer {peer_id}...")

        first_msg_received = False  # the first message should come quickly, if not, we lost the peer
        # handle websocket messages and loop for lifetime
        while True:
            try:
                if self.pcs.get(peer_id) and (
                    self.pcs[peer_id].connectionState == "connected"
                    or self.pcs[peer_id].connectionState == "disconnected"
                    or self.pcs[peer_id].connectionState == "failed"
                ):
                    await q.put.aio("close", partition="server")
                    break

                # get websocket message and parse as json
                msg = json.loads(
                    await asyncio.wait_for(
                        q.get.aio(partition=peer_id), timeout=5
                    )
                )

                first_msg_received = True

                # handle offer
                if msg.get("type") == "offer":
                    print(f"Peer {self.id} received offer from {peer_id}...")

                    await self._handle_offer(peer_id, msg)

                    # generate and send answer
                    await q.put.aio(
                        json.dumps(self._generate_answer(peer_id)),
                        partition="server",
                    )

                # handle ice candidate (trickle ice)
                elif msg.get("type") == "ice_candidate":
                    candidate = msg.get("candidate")

                    if not candidate or not self.pcs.get(peer_id):
                        return

                    print(
                        f"Peer {self.id} received ice candidate from {peer_id}..."
                    )

                    # parse ice candidate
                    ice_candidate = candidate_from_sdp(
                        candidate["candidate_sdp"]
                    )
                    ice_candidate.sdpMid = candidate["sdpMid"]
                    ice_candidate.sdpMLineIndex = candidate["sdpMLineIndex"]

                    await self._handle_ice_candidate(peer_id, ice_candidate)

                # get peer's id
                elif msg.get("type") == "identify":
                    await q.put.aio(
                        json.dumps({"type": "identify", "peer_id": self.id}),
                        partition="server",
                    )

                elif msg.get("type") == "get_turn_servers":
                    print(f"Sending turn servers to peer {peer_id}...")
                    await q.put.aio(
                        json.dumps(self.get_turn_servers()), partition="server"
                    )

                else:
                    print(f"Unknown message type: {msg.get('type')}")

            except Exception as e:
                if isinstance(e, TimeoutError):
                    if not first_msg_received:
                        print(
                            f"Modal peer instance for client peer {peer_id} lost connection to client"
                        )
                        return
                continue

        print(
            f"Modal peer instance for client peer {peer_id} connected, running streams..."
        )
        await self._run_streams(peer_id)

        print(f"Shutting down modal peer instance for client peer {peer_id}...")

    async def _setup_peer_connection(self, peer_id):
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection

        # create peer connection with STUN server
        # aiortc automatically uses google's STUN server when
        # self.pcs[peer_id] = RTCPeerConnection()
        # is called, but we can also specify our own:
        config = RTCConfiguration(
            [RTCIceServer(urls="stun:stun.l.google.com:19302")]
        )
        self.pcs[peer_id] = RTCPeerConnection(configuration=config)

        await self.setup_streams(peer_id)

        print(
            f"Created peer connection and setup streams from {self.id} to {peer_id}"
        )

    async def setup_streams(self, peer_id):
        """
        Any custom logic when setting up the connection and streams
        """
        pass

    def get_turn_servers(self):
        """
        Returns a list of TURN servers
        """
        pass

    async def _generate_offer(self, peer_id):
        print(f"Peer {self.id} generating offer for {peer_id}...")

        # initalize peer connection
        await self._setup_peer_connection(peer_id)
        # create initial offer
        offer = await self.pcs[peer_id].createOffer()
        # set local/our description, this also triggers and waits for ICE gathering/generation of ICE candidate info
        await self.pcs[peer_id].setLocalDescription(offer)
        # NOTE: we can't use `offer.sdp` because the ICE candidates are not included
        # these are embedded in the SDP after setLocalDescription() is called
        return {
            "sdp": self.pcs[peer_id].localDescription.sdp,
            "type": offer.type,
            "peer_id": self.id,
        }

    async def _handle_offer(self, peer_id, offer):
        from aiortc import RTCSessionDescription

        print(f"Peer {self.id} handling offer from {peer_id}...")

        # initalize peer connection and streams
        await self._setup_peer_connection(peer_id)
        # set remote description
        await self.pcs[peer_id].setRemoteDescription(
            RTCSessionDescription(offer["sdp"], offer["type"])
        )
        # create answer
        answer = await self.pcs[peer_id].createAnswer()
        # set local/our description, this also triggers ICE gathering
        await self.pcs[peer_id].setLocalDescription(answer)

    def _generate_answer(self, peer_id):
        print(f"Peer {self.id} generating answer for {peer_id}...")

        return {
            "sdp": self.pcs[peer_id].localDescription.sdp,
            "type": "answer",
            "peer_id": self.id,
        }

    async def _handle_answer(self, peer_id, answer):
        from aiortc import RTCSessionDescription

        print(f"Peer {self.id} handling answer from {peer_id}...")
        # set remote peer description
        await self.pcs[peer_id].setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )

    async def _handle_ice_candidate(self, peer_id, candidate):
        import asyncio

        print(f"Peer {self.id} handling ice candidate from {peer_id}...")

        # sometimes this event is called before
        # the peer connection is created on this end
        retries = 5
        while not self.pcs.get(peer_id) and retries > 0:
            await asyncio.sleep(0.1)
            retries -= 1

        if not retries:
            print(
                f"Peer {self.id} failed to create peer connection for {peer_id} before ICE candidate event"
            )
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
