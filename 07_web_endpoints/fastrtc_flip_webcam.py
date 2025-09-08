# ---
# cmd: ["modal", "serve", "07_web_endpoints/fastrtc_flip_webcam.py"]
# deploy: true
# ---

# # Run a FastRTC app on Modal

# [FastRTC](https://fastrtc.org/) is a Python library for real-time communication on the web.
# This example demonstrates how to run a simple FastRTC app in the cloud on Modal.

# It's intended to help you get up and running with real-time streaming applications on Modal
# as quickly as possible. If you're interested in running a production-grade WebRTC app on Modal,
# see [this example](https://modal.com/docs/examples/webrtc_yolo).

# In this example, we stream webcam video from a browser to a container on Modal,
# where the video is flipped, annotated, and sent back with under 100ms of delay.
# You can try it out [here](https://modal-labs-examples--example-fastrtc-flip-webcam-ui.modal.run/)
# or just dive straight into the code to run it yourself.

# ## Set up FastRTC on Modal

# First, we import the `modal` SDK
# and use it to define a [container image](https://modal.com/docs/guide/images)
# with FastRTC and related dependencies.

import modal

web_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]==0.115.4",
    "fastrtc==0.0.23",
    "gradio==5.7.1",
    "opencv-python-headless==4.11.0.86",
)

# Then, we set that as the default Image on our Modal [App](https://modal.com/docs/guide/apps).

app = modal.App("example-fastrtc-flip-webcam", image=web_image)

# ### Configure WebRTC streaming on Modal

# Under the hood, FastRTC uses the WebRTC
# [APIs](https://www.w3.org/TR/webrtc/) and
# [protocols](https://datatracker.ietf.org/doc/html/rfc8825).

# WebRTC provides low latency ("real-time") peer-to-peer communication
# for Web applications, focusing on audio and video.
# Considering that the Web is a platform originally designed
# for high-latency, client-server communication of text and images,
# that's no mean feat!

# In addition to protocols that implement this communication,
# WebRTC includes APIs for describing and manipulating audio/video streams.
# In this demo, we set a few simple parameters, like the direction of the webcam
# and the minimum frame rate. See the
# [MDN Web Docs for `MediaTrackConstraints`](https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackConstraints)
# for more.

TRACK_CONSTRAINTS = {
    "width": {"exact": 640},
    "height": {"exact": 480},
    "frameRate": {"min": 30},
    "facingMode": {  # https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackSettings/facingMode
        "ideal": "user"
    },
}

# In theory, the Internet is designed for peer-to-peer communication
# all the way down to its heart, the Internet Protocol (IP): just send packets between IP addresses.
# In practice, peer-to-peer communication on the contemporary Internet is fraught with difficulites,
# from restrictive firewalls to finicky work-arounds for
# [the exhaustion of IPv4 addresses](https://www.a10networks.com/glossary/what-is-ipv4-exhaustion/),
# like [Carrier-Grade Network Address Translation (CGNAT)](https://en.wikipedia.org/wiki/Carrier-grade_NAT).

# So establishing peer-to-peer connections can be quite involved.
# The protocol for doing so is called Interactive Connectivity Establishment (ICE).
# It is described in [this RFC](https://datatracker.ietf.org/doc/html/rfc8445#section-2).

# ICE involves the peers exchanging a list of connections that might be used.
# We use a fairly simple setup here, where our peer on Modal uses the
# [Session Traversal Utilities for NAT (STUN)](https://datatracker.ietf.org/doc/html/rfc5389)
# server provided by Google. A STUN server basically just reflects back to a client what their
# IP address and port number appear to be when they talk to it. The peer on Modal communicates
# that information to the other peer trying to connect to it -- in this case, a browser trying to share a webcam feed.
# Note the use of `stun` and port `19302` in the URL in place of
# something more familiar, like `http` and port `80`.

RTC_CONFIG = {"iceServers": [{"url": "stun:stun.l.google.com:19302"}]}


# ## Running a FastRTC app on Modal

# FastRTC builds on top of the [Gradio](https://www.gradio.app/docs)
# library for defining Web UIs in Python.
# Gradio in turn is compatible with the
# [Asynchronous Server Gateway Interface (ASGI)](https://asgi.readthedocs.io/en/latest/)
# protocol for asynchronous Python web servers, like
# [FastAPI](https://fastrtc.org/userguide/streams/),
# so we can host it on Modal's cloud platform using the
# [`modal.asgi_app` decorator](https://modal.com/docs/guide/webhooks#serving-asgi-and-wsgi-apps)
# with [Modal Function](https://modal.com/docs/guide/apps).

# But before we do that, we need to consider limits:
# on how many peers can connect to one instance on Modal
# and on how long they can stay connected.
# We picked some sensible defaults to show how they interact
# with the deployment parameters of the Modal Function.
# You'll want to tune these for your application!

MAX_CONCURRENT_STREAMS = 10  # number of peers per instance on Modal

MINUTES = 60  # seconds
TIME_LIMIT = 10 * MINUTES  # time limit


@app.function(
    # gradio requires sticky sessions
    # so we limit the number of concurrent containers to 1
    # and allow that container to handle concurrent streams
    max_containers=1,
    scaledown_window=TIME_LIMIT + 1 * MINUTES,  # add a small buffer to time limit
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_STREAMS)  # inputs per container
@modal.asgi_app()  # ASGI on Modal
def ui():
    import fastrtc  # WebRTC in Gradio
    import gradio as gr  # WebUIs in Python
    from fastapi import FastAPI  # asynchronous ASGI server framework
    from gradio.routes import mount_gradio_app  # connects Gradio and FastAPI

    with gr.Blocks() as blocks:  # block-wise UI definition
        gr.HTML(  # simple HTML header
            "<h1 style='text-align: center'>"
            "Streaming Video Processing with Modal and FastRTC"
            "</h1>"
        )

        with gr.Column():  # a column of UI elements
            fastrtc.Stream(  # high-level media streaming UI element
                modality="video",
                mode="send-receive",
                handler=flip_vertically,  # handler -- handle incoming frame, produce outgoing frame
                ui_args={"title": "Click 'Record' to flip your webcam in the cloud"},
                rtc_configuration=RTC_CONFIG,
                track_constraints=TRACK_CONSTRAINTS,
                concurrency_limit=MAX_CONCURRENT_STREAMS,  # limit simultaneous connections
                time_limit=TIME_LIMIT,  # limit time per connection
            )

    return mount_gradio_app(app=FastAPI(), blocks=blocks, path="/")


# To try this out for yourself, run

# ```bash
# modal serve 07_web_endpoints/fastrtc_flip_webcam.py
# ```

# and head to the `modal.run` URL that appears in your terminal.
# You can also check on the application's dashboard
# via the `modal.com` URL thatappears below it.

# The `modal serve` command produces a hot-reloading development server --
# try editing the `title` in the `ui_args` above and watch the server redeploy.

# This temporary deployment is tied to your terminal session.
# To deploy permanently, run

# ```bash
# modal deploy 07_web_endponts/fastrtc_flip_webcam.py
# ```

# Note that Modal is a serverless platform with [usage-based pricing](https://modal.com/pricing),
# so this application will spin down and cost you nothing when it is not in use.

# ## Addenda

# This FastRTC app is very much the "hello world" or "echo server"
# of FastRTC: it just flips the incoming webcam stream and adds a "hello" message.
# That logic appears below.


def flip_vertically(image):
    import cv2
    import numpy as np

    image = image.astype(np.uint8)

    if image is None:
        print("failed to decode image")
        return

    # flip vertically and caption to show video was processed on Modal
    image = cv2.flip(image, 0)
    lines = ["Hello from Modal!"]
    caption_image(image, lines)

    return image


def caption_image(
    img, lines, font_scale=0.8, thickness=2, margin=10, font=None, color=None
):
    import cv2

    if font is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
    if color is None:
        color = (127, 238, 100, 128)  # Modal Green

    # get text sizes
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    if not sizes:
        return

    # position text in bottom right
    pos_xs = [img.shape[1] - size[0] - margin for size in sizes]

    pos_ys = [img.shape[0] - margin]
    for _width, height in reversed(sizes[:-1]):
        next_pos = pos_ys[-1] - 2 * height
        pos_ys.append(next_pos)

    for line, pos in zip(lines, zip(pos_xs, reversed(pos_ys))):
        cv2.putText(img, line, pos, font, font_scale, color, thickness)
