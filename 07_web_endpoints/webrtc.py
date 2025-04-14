# from fastrtc import Stream
# import numpy as np
# from fastapi import FastAPI
# from fastapi.responses import HTMLResponse

# def flip_vertically(image):
#     return np.flip(image, axis=0)


# stream = Stream(
#     handler=flip_vertically,
#     modality="video",
#     mode="send-receive",
# )

# app = FastAPI()
# stream.mount(app)

# # Optional: Add routes
# @app.get("/")
# async def _():
#     return HTMLResponse(content=open("index.html").read())

# # uvicorn app:app --host 0.0.0.0 --port 8000

import numpy as np
import gradio as gr
from fastrtc import WebRTC, Stream

def flip_vertically(image):
    return np.flip(image, axis=0)





with gr.Blocks() as demo:
    gr.HTML(
    """
    <h1 style='text-align: center'>
    Chat (Powered by WebRTC ⚡️)
    </h1>
    """
    )
    with gr.Column():

        stream = Stream(
            handler=flip_vertically,
            modality="video",
            mode="send-receive",
        )
demo.launch()