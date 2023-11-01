# Migrating an async video processing job from a Celery queue to Modal

# This example shows the migration of a small project from a Celery job queue deployed on Render to Modal

# ## Background on the application

# This application automatically converts videos to Markdown + screenshot tutorials. The video inputs typically range from 3-5 minutes and are casual demo videos/Looms made by product managers.
# Since there is heavy video processing involved (transcribing the video, parsing through it in search of certain timeframes), we want to be able to offload all of this to a separate job queue, so that the user doesn't have to wait for the video to be processed before they can continue using the application.

# ## The original application

# The original application was a Flask app that accepted a video file upload, and then used Celery to queue up the video processing job. The video processing job would upload the video to S3, extract the relevant timestamps/screenshots, and upload those to Supabase.

# ```python
# app.py
#
# import os
# import uuid
# from flask import Flask, request, jsonify, redirect, flash
# from flask_cors import CORS
# import json
# from celery.result import AsyncResult
# from tasks import extract_screenshots, extract_screenshots_from_video_and_upload_celery, transcribe_video_and_extract_screenshots
# from tasks import app as celery_app
# from gcs_utilities import upload_video_to_gcs
# from database_utilities import insert_project

# app = Flask(__name__)
# CORS(app)

# @app.route("/upload", methods=["POST"])
# def upload():

#     # get the title from the form
#     title = request.form.get("title")
#     # get the video from the form
#     video = request.files.get("video")
#     # get the user_id from the form
#     user_id = request.form.get("user_id")

#     # generate random project id
#     project_id = uuid.uuid4()

#     # upload the video to google cloud storage
#     video_url, folder_name = upload_video_to_gcs(project_id, title, video)

#     # create a project entry in the database
#     insert_project(project_id, user_id, 0, video_url, title, folder_name)

#     # send the video processing task to the Celery queue
#     result = transcribe_video_and_extract_screenshots.delay(project_id, video_url, title)

#     return {"project_id": project_id}
# ```


# ```python
# # tasks.py
#
# import os
# from celery import Celery
# from celery.utils.log import get_task_logger
# from video_utilities import transcribe_video_whisper
# from database_utilities import fetch_project_data, update_project_status, insert_timestamps_and_text

# app = Celery('tasks', broker=os.getenv("CELERY_BROKER_URL"), backend=os.getenv("CELERY_RESULT_BACKEND"))
# logger = get_task_logger(__name__)

# @app.task
# def transcribe_video_and_extract_screenshots(project_id, video_url, title):

#     # fetch the project data from supabase
#     project_data = fetch_project_data(project_id)

#     # check the status of the project
#     if project_data["task_status"] == 1:
#         return

#     # transcribe the video
#     sentences = transcribe_video_whisper(video_url)

#     timestamps_and_text = extract_screenshots(sentences)

#     # insert the timestamps and text into the database
#     insert_timestamps_and_text(project_id, timestamps_and_text["phrase_texts"], timestamps_and_text["relevant_frames"])

#     # update the status of the project to be "completed"
#     update_project_status(project_id, 1)

#     return
# ```

# The Celery queue was working fine, but it still took on the order of ~30 seconds to process a 2-3 minute video. I was hoping to be able to speed this up, and decided to turn to Modal.
# Some addition requirements: I wanted to minimize the amount of refactoring, and wanted to continue to host the Flask app on Render. All I wanted was to be able to offload the video processing work to a job somewhere, and for that work to complete faster.

# ## The Modal application

# To use Modal instead of a Celery queue, we first create a separate file, `modal_queue.py`, and define a Modal Stub and Image.
# The image is a Debian image with ffmpeg, wget, git, and the python dependencies installed.

# We also import the helper function from `tasks.py`, `transcribe_video_and_extract_screenshots`, which is doing the bulk of the video processing work.
# Note that we are importing `transcribe_video_extract_screenshots` from outside of the Modal function. This is because the `tasks.py` file exists locally.

import modal
from tasks import transcribe_video_and_extract_screenshots

FACE_CASCADE_FN = "haarcascade_frontalface_default.xml"

stub = modal.Stub(
    "process-video-job",
    image=modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "wget", "git", "ffmpeg")
    .run_commands(
        f"wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{FACE_CASCADE_FN} -P /root"
    )
    .pip_install(
        "stable-ts==2.6.2",
        "supabase==1.0.3",
        "openai==0.27.2",
        "opencv-python==4.7.0.72",
        "google-api-core==2.11.0",
        "google-auth==2.16.3",
        "google-cloud-core==2.3.2",
        "google-cloud-storage==2.7.0",
        "google-crc32c==1.5.0",
        "google-resumable-media==2.4.1",
    ),
)

# Next, we define a function, `process_video`, and annotate it with the `@stub.function` decorator, which tells
# Modal to run this function on the stub we defined above. We also specify that this function should run on any GPU, and give it access to certain environment secrets.
@stub.function(
    secret=modal.Secret.from_name("video-to-tutorial-keys"),
    gpu="any",
    retries=3,
)
def process_video(project_id: str, video_url: str, title: str):

    transcribe_video_and_extract_screenshots(project_id, video_url, title)


# With our `process_video` function defined, we can now deploy our Stub to Modal by running `modal deploy modal_queue.py` in the terminal. This will build the image, push it to the Modal registry, and deploy the stub to Modal.

# We can now see the `process_video` function in the Modal dashboard, where we can also access logs, performance benchmarks, etc.

# ![modal interface function](./modal-interface-functions.png)

# Now we can come back to our original Flask app and change the API endpoint to call the `process_video` Modal function instead of the Celery queue.
# We call the function using [`.spawn`](https://modal.com/docs/reference/modal.Function#spawn), which means that we don't have to wait for the results. Instead, it returns a `FunctionCalls` object, similar to a future/promise, which can be polled on later.

import uuid

from database_utilities import insert_project
from flask import Flask, request
from flask_cors import CORS
from gcs_utilities import upload_video_to_gcs
from modal import Function

app = Flask(__name__)
CORS(app)


@app.route("/upload", methods=["POST"])
def upload():

    # get the title from the form
    title = request.form.get("title")
    # get the video from the form
    video = request.files.get("video")
    # get the user_id from the form
    user_id = request.form.get("user_id")

    # generate random project id
    project_id = uuid.uuid4()

    # upload the video to google cloud storage
    video_url, folder_name = upload_video_to_gcs(project_id, title, video)

    # create a project entry in the database
    insert_project(project_id, user_id, 0, video_url, title, folder_name)

    # Lookup the modal function and call it. Note that `process-video-job` is the name on the Stub above, while `process_video` is the name of the function we defined above.
    fn = Function.lookup("process-video-job", "process_video")
    fn.spawn(project_id, video_url, title)

    return {"project_id": project_id}


# Finally, when we go to redeploy the Flask app on Render, the only changes we need to make to the deployment environment are to include the Modal client in the `requirements.txt` file and the Modal API TOKEN_SECRET and TOKEN_ID in the environment variables.

# On Modal, a 2-3 minute video takes ~40 seconds to process.
