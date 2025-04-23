import modal

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("python3-opencv", "ffmpeg")
    .run_commands("pip install --upgrade pip")
    .pip_install(
        "fastapi[standard]==0.115.4",
        "gradio~=5.7.1",
        "fastrtc",
        "opencv-python",
    )
)

MAX_CONCURRENT_STREAMS = 10

app = modal.App(
    "fastrtc-webcam-demo",
    image=web_image,
)

@app.cls(
    image=web_image,
    # gradio requires sticky sessions
    # so we limit the number of concurrent containers to 1
    # and allow it to scale to 100 concurrent inputs
    max_containers=1,
    # region="ap-south"
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_STREAMS) # we also need to set the limit on the fastrtc stream
class WebRTCApp:

    @modal.enter()
    def init(self):
        self.last_frame_time = None
    
    @modal.asgi_app()
    def ui(self):

        import time
        import numpy as np
        import cv2

        from fastapi import FastAPI
        import gradio as gr
        from gradio.routes import mount_gradio_app

        from fastrtc import Stream

        def flip_vertically(image):

            now = time.time()
            if self.last_frame_time is None:
                round_trip_time = np.nan
            else:
                round_trip_time = now - self.last_frame_time
            self.last_frame_time = now

            img = image.astype(np.uint8)
                    
            if img is None:
                print("Failed to decode image")
                return None
                        
            # Flip vertically
            flipped = cv2.flip(img, 0)

            # add round trip time to image
            text1 = "Round trip time:"
            text2 = f"{round_trip_time*1000:>6.1f} msec"
            font_scale = 0.8
            thickness = 2
            
            # Get text sizes
            (text1_width, text1_height), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            (text2_width, text2_height), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Position text in bottom right, with text2 below text1
            margin = 10
            text1_x = flipped.shape[1] - text1_width - margin
            text1_y = flipped.shape[0] - text1_height - margin - text2_height
            text2_x = flipped.shape[1] - text2_width - margin  
            text2_y = flipped.shape[0] - margin

            # Draw both lines of text
            cv2.putText(flipped, text1, (text1_x, text1_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0, 128), thickness)
            cv2.putText(flipped, text2, (text2_x, text2_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0, 128), thickness)

            return flipped

        with gr.Blocks() as blocks:
            gr.HTML(
            """
            <h1 style='text-align: center'>
            Streaming Video Processing with Modal and FastRTC
            </h1>
            """
            )
            with gr.Column():

                stream = Stream(
                    handler=flip_vertically,
                    modality="video",
                    mode="send-receive",
                    rtc_configuration={
                        "iceServers": [{"url": "stun:stun.l.google.com:19302"}]
                    },
                    ui_args={
                        "title": "Click Record to Flip Your Webcam in the Cloud",
                    },
                    track_constraints= {
                        "width": {"exact": 640},
                        "height": {"exact": 480},
                        "frameRate": {"min": 30},
                        "facingMode": {"ideal": "environment"},
                    },
                    concurrency_limit=MAX_CONCURRENT_STREAMS
                )
            
        return mount_gradio_app(app=FastAPI(), blocks=blocks, path="/")