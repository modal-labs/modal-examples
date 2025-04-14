import modal

web_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]==0.115.4",
    "gradio~=5.7.1",
    "pillow~=10.2.0",
    "fastrtc",
)

app = modal.App(
    "webrtc-video-demo",
    image=web_image,
)

rtc_config = {
    "iceServers": [{"url": "stun:stun.l.google.com:19302"}]
}

@app.cls(
    image=web_image,
    min_containers=1,
    scaledown_window=60 * 20,
    # gradio requires sticky sessions
    # so we limit the number of concurrent containers to 1
    # and allow it to scale to 100 concurrent inputs
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
class WebRTCApp:
    
    @modal.asgi_app()
    def ui(self):

        import numpy as np
        from fastrtc import Stream
        import gradio as gr
        from gradio.routes import mount_gradio_app
        from fastapi import FastAPI

        def flip_vertically(image):
            return np.flip(image, axis=0)

        with gr.Blocks() as blocks:
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
                    rtc_configuration=rtc_config,
                )
            
        return mount_gradio_app(app=FastAPI(), blocks=blocks, path="/")