# ---
# mypy: ignore-errors
# ---

# # Build a sessioned chat room
#
# This example uses [Sticky Sessions](https://modal.com/docs/guide/sticky-sessions)
# to create a chat room where every client within the same session is routed to the same container.
# As a result, messages are broadcast to all clients part of that session.
#
# Deploy the Server before starting or connecting to a session:
#
# ```bash
# modal deploy sessioned_chatroom.py
# ```

import argparse
import asyncio
import sys

import aiohttp
import modal

app = modal.App("example-sessioned-chatroom")

chat_image = modal.Image.debian_slim().uv_pip_install("fastapi[standard]==0.115.4")


# ## Broadcast messages within each session
#
# A container can host multiple sessions, so the Server keeps a separate connection
# set for each session ID. The Server sends each message to every active connection in that
# session.


@app.server(image=chat_image, unauthenticated=True)
@modal.sessioned()
class ChatRoom:
    @modal.enter()
    def start(self):
        import threading

        import uvicorn
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import PlainTextResponse

        self.rooms: dict[str, set[WebSocket]] = {}
        web_app = FastAPI()

        @web_app.get("/")
        async def index():
            return PlainTextResponse("OK")

        @web_app.websocket("/ws")
        async def chat(ws: WebSocket):
            await ws.accept()
            session_id = ws.headers["x-modal-server-session-id"]
            connections = self.rooms.setdefault(session_id, set())
            connections.add(ws)

            try:
                while True:
                    message = await ws.receive_text()
                    for connection in list(connections):
                        try:
                            await connection.send_text(message)
                        except Exception:
                            pass
            except WebSocketDisconnect:
                pass
            finally:
                connections.discard(ws)
                if not connections:
                    del self.rooms[session_id]

        threading.Thread(
            target=uvicorn.run,
            kwargs={"app": web_app, "host": "0.0.0.0", "port": 8000},
            daemon=True,
        ).start()


# ## Start or connect to a room
#
# Start a session to create a room and print its token:
#
# ```bash
# python sessioned_chatroom.py --start
# ```
#
# Anyone with that token can connect to the same room. Run this command in multiple
# terminals to chat between clients:
#
# ```bash
# python sessioned_chatroom.py --connect <token>
# ```


async def read_input():
    loop = asyncio.get_running_loop()
    line = loop.create_future()
    fileno = sys.stdin.fileno()

    def read_line():
        loop.remove_reader(fileno)
        line.set_result(sys.stdin.readline())

    print("> ", end="", flush=True)
    loop.add_reader(fileno, read_line)
    try:
        return await line
    finally:
        loop.remove_reader(fileno)


async def send_messages(ws):
    while line := await read_input():
        await ws.send_str(line.rstrip("\n"))
    await ws.close()


async def receive_messages(ws):
    async for message in ws:
        print(message.data)


async def connect(server, token):
    ws_url = (await server.get_url.aio()).replace("https://", "wss://") + "/ws"
    headers = {"Modal-Authorization": f"Bearer {token}"}
    async with aiohttp.ClientSession(headers=headers) as http_session:
        async with http_session.ws_connect(ws_url) as ws:
            tasks = {
                asyncio.create_task(send_messages(ws)),
                asyncio.create_task(receive_messages(ws)),
            }
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()


def parse_args():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--start", action="store_true", help="start a session and print its token"
    )
    mode.add_argument("--connect", metavar="TOKEN", help="connect to a session")
    return parser.parse_args()


async def main(args):
    server = modal.Server.from_name("example-sessioned-chatroom", "ChatRoom")
    if args.start:
        session = await server.sessions.start.aio(idle_timeout=3600)
        print(session.token)
    else:
        await connect(server, args.connect)


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        pass


# ## Testing
#
# To quickly test that our implementation works, run:
#
# ```bash
# modal run sessioned_chatroom.py
# ```


@app.local_entrypoint()
async def smoke_test():
    session = await ChatRoom.sessions.start.aio(idle_timeout=60)
    try:
        ws_url = (await ChatRoom.get_url.aio()).replace("https://", "wss://") + "/ws"
        headers = {"Modal-Authorization": f"Bearer {session.token}"}
        async with aiohttp.ClientSession(headers=headers) as http_session:
            async with http_session.ws_connect(ws_url) as bob:
                async with http_session.ws_connect(ws_url) as alice:
                    await alice.send_str("hello")
                    message = await bob.receive(timeout=30)
                    assert message.data == "hello", message
    finally:
        await ChatRoom.sessions.terminate.aio(session.token)
