# ---
# deploy: true
# cmd: ["modal", "serve", "07_web_endpoints/count_faces.py"]
# ---

# # Run OpenCV face detection on an image

# This example shows how you can use OpenCV on Modal to detect faces in an image. We use
# the `opencv-python` package to load the image and the `opencv` library to
# detect faces. The function `count_faces` takes an image as input and returns
# the number of faces detected in the image.

# The code below also shows how you can create a web server with Sanic to upload
# an image and get the number of faces detected in the image.

import os

import modal

app = modal.App("example-count-faces")


open_cv_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("python3-opencv")
    .pip_install("opencv-python~=4.10.0", "numpy<2")
)


@app.function(image=open_cv_image)
def count_faces(image_bytes):
    import cv2
    import numpy as np

    # Example borrowed from https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(
        os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
    )
    # Read the input image
    np_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces)


@app.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "sanic~=24.6.0"
    )
)
@modal.asgi_app()
def web():
    import sanic

    app = sanic.Sanic("web_worker_example")

    @app.get("/")
    def index(request):
        return sanic.html(
            """
            <html>
            <form action="/process" method="post" enctype="multipart/form-data">
                <input type="file" name="file" id="file" />
                <input type="submit" />
            </form>
            </html>
            """
        )

    @app.post("/process")
    async def process(request: sanic.Request):
        input_file = request.files["file"][0]
        async with app.run():  # type: ignore
            num_faces = await count_faces.remote(input_file.body)

        return sanic.json({"faces": num_faces})

    return app
