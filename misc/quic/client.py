# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "aioquic==1.2.0",
#     "aiohttp==3.12.13",
#     "opencv-python==4.11.0.86",
#     "pynat==0.6.0",
# ]
# ///

import argparse
import asyncio
import socket
import ssl
import time
import uuid

import aiohttp
import cv2
import numpy as np
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration
from pynat import get_stun_response


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QUIC-YOLO webcam client")
    p.add_argument("--url", type=str, required=True, help="Rendezvous URL")
    p.add_argument(
        "--local-port", type=int, default=5555, help="Local UDP port to bind."
    )
    p.add_argument(
        "--max-fps",
        type=float,
        default=30.0,
        help="Maximum FPS for real-time video streaming (higher = smoother but more bandwidth)",
    )
    p.add_argument(
        "--fake", action="store_true", help="Send synthetic video instead of webcam"
    )
    return p.parse_args()


async def get_ext_addr(sock: socket.socket) -> tuple[str, int]:
    resp = get_stun_response(sock, ("stun.ekiga.net", 3478))
    return resp["ext_ip"], resp["ext_port"]


async def run(args):
    client_id = str(uuid.uuid4())

    # discover public mapping via STUN
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.local_port))
    sock.setblocking(False)

    pub_ip, pub_port = await get_ext_addr(sock)
    print(f"Public tuple: {pub_ip}:{pub_port}")

    # register & wait for the peer's tuple
    async with aiohttp.ClientSession() as session:
        while True:
            resp = await session.post(
                f"{args.url}/register",
                json={
                    "role": "client",
                    "peer_id": client_id,
                    "ip": pub_ip,
                    "port": pub_port,
                },
            )
            peer = (await resp.json()).get("peer")
            if peer:
                peer_ip, peer_port = peer
                break
            await asyncio.sleep(1)
    print(f"Peer tuple: {peer_ip}:{peer_port}")

    for _ in range(150):  # 15s total (for cold-start latency)
        sock.sendto(b"punch", (peer_ip, peer_port))
        try:
            await asyncio.wait_for(asyncio.get_event_loop().sock_recv(sock, 16), 0.05)
            break
        except asyncio.TimeoutError:
            continue
    else:
        raise RuntimeError("NAT hole punching failed – no response from server")
    print(f"Punched {pub_ip}:{pub_port} -> {peer_ip}:{peer_port}")

    sock.close()  # close socket, mapping should stay alive

    cfg = QuicConfiguration(
        is_client=True,
        alpn_protocols=["hq-29"],
        verify_mode=ssl.CERT_NONE,
    )

    async with connect(
        peer_ip,
        peer_port,
        configuration=cfg,
        local_port=args.local_port,
    ) as quic:
        reader, writer = await quic.create_stream()

        cap = None
        if not args.fake:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print("[WARN] Webcam not available. Falling back to synthetic frames.")
                cap.release()
                cap = None

        frame_h, frame_w = 480, 640

        def gen_fake_frame(
            idx: int,
        ) -> np.ndarray:  # generate a simple synthetic BGR frame with moving rectangle
            img = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            size = 80
            x = (idx * 5) % (frame_w - size)
            y = (idx * 3) % (frame_h - size)
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + size, y + size), color, -1)
            cv2.putText(
                img,
                "SYNTH",
                (10, frame_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            return img

        min_frame_interval = 1.0 / args.max_fps
        last_frame_ts = 0.0
        frame_idx = 0

        try:
            while True:
                ts = time.time()
                if ts - last_frame_ts < min_frame_interval:
                    await asyncio.sleep(min_frame_interval - (ts - last_frame_ts))
                last_frame_ts = time.time()

                if cap is not None:
                    ret, frame = cap.read()
                    if not ret:
                        print(
                            "Failed to read frame from webcam; switching to synthetic."
                        )
                        cap.release()
                        cap = None
                        frame = gen_fake_frame(frame_idx)
                else:
                    frame = gen_fake_frame(frame_idx)

                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    print("JPEG encode failed")
                    continue
                data = buf.tobytes()

                # send frame
                writer.write(len(data).to_bytes(4, "big") + data)
                await writer.drain()

                # receive annotated frame
                hdr = await reader.readexactly(4)
                resp_len = int.from_bytes(hdr, "big")
                if resp_len == 0:
                    print("Server closed stream")
                    break
                resp_bytes = await reader.readexactly(resp_len)
                img_np = np.frombuffer(resp_bytes, dtype=np.uint8)
                annotated = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                if annotated is None:
                    continue

                frame_idx += 1

                cv2.imshow("YOLO (Remote)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            writer.write((0).to_bytes(4, "big"))
            await writer.drain()
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
