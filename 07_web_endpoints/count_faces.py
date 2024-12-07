# ---
# cmd: ["modal", "serve", "07_web_endpoints/count_faces.py"]
# ---

# # Run OpenCV face detection on an image

# This example shows how you can use OpenCV on Modal to detect faces in an image. We use
# the `opencv-python` package to load the image and the `opencv` library to
# detect faces. The function `count_faces` takes an image as input and returns
# the number of faces detected in the image.

# The code below also shows how you can create wrap this function
# in a simple FastAPI server to create a web interface.

import os

import modal

app = modal.App("example-count-faces")


open_cv_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("python3-opencv")
    .pip_install(
        "fastapi[standard]==0.115.4",
        "opencv-python~=4.10.0",
        "numpy<2",
    )
)


@app.function(image=open_cv_image)
def count_faces(image_bytes: bytes) -> int:
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
    image=modal.Image.debian_slim(python_version="3.11").pip_install("inflect")
)
@modal.asgi_app()
def web():
    import inflect
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """
        Render an HTML form for file upload.
        """
        return """
        <html>
            <head>
                <title>Face Counter</title>
            </head>
            <body>
                <h1>Upload an Image to Count Faces</h1>
                <form action="/process" method="post" enctype="multipart/form-data">
                    <input type="file" name="file" id="file" accept="image/*" required />
                    <button type="submit">Upload</button>
                </form>
            </body>
        </html>
        """

    @app.post("/process", response_class=HTMLResponse)
    async def process(file: UploadFile = File(...)):
        """
        Process the uploaded image and return the number of faces detected.
        """
        try:
            file_content = await file.read()
            num_faces = await count_faces.remote.aio(file_content)
            return f"""
            <html>
                <head>
                    <title>Face Counter Result</title>
                </head>
                <body>
                    <h1>{inflect.engine().number_to_words(num_faces).title()} {'Face' if num_faces==1 else 'Faces'} Detected</h1>
                    <h2>{"😀" * num_faces}</h2>
                    <a href="/">Go back</a>
                </body>
            </html>
            """
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error processing image: {str(e)}"
            )

    return app
