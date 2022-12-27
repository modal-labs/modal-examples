# ---
# deploy: true
# output-directory: "/tmp"
# ---
# # Face detection on YouTube videos
#
# This is an example that uses
# [OpenCV](https://github.com/opencv/opencv-python)
# as well as video utilities
# [pytube](https://pytube.io/en/latest/)
# and
# [moviepy](https://zulko.github.io/moviepy/)
# to work with video files.
#
# The face detection is a quite simple model built into OpenCV
# and is not state of the art.
#
# ## The result
#
# <center>
# <video controls>
# <source src="./youtube_face_detection.mp4" type="video/mp4">
# <track kind="captions" />
# </video>
# </center>
#
# If you watched this, we succeeded
# [rickrolling](https://en.wikipedia.org/wiki/Rickrolling)
# you! 🤣
#
#
# ## The Python code
#
# We start by setting up the container image we need.
# This requires installing a few dependencies needed for OpenCV as well as downloading the face detection model

import os

import modal

OUTPUT_DIR = "/tmp/"
FACE_CASCADE_FN = "haarcascade_frontalface_default.xml"

image = (
    modal.Image.debian_slim()
    .run_commands(
        [
            "apt-get install -y libgl1-mesa-glx libglib2.0-0 wget",
            f"wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{FACE_CASCADE_FN} -P /root",
        ]
    )
    .pip_install(["pytube", "opencv-python", "moviepy"])
)
stub = modal.Stub("example-youtube-face-detection", image=image)

if stub.is_inside():
    import cv2
    import moviepy.editor
    import pytube

# For temporary storage of movie clips, we use a "shared volume"

stub.sv = modal.SharedVolume()

# ### Face detection function
#
# The face detection function takes three arguments:
#
# * A filename to the source clip
# * A time slice denoted by start and a stop in seconds
#
# The function extracts the subclip from the movie file (which is stored on the shared volume),
# runs face detection on every frame in its slice,
# and stores the resulting video back to the shared storage.


@stub.function(shared_volumes={"/clips": stub.sv}, timeout=600)
def detect_faces(fn, start, stop):
    # Extract the subclip from the video
    clip = moviepy.editor.VideoFileClip(fn).subclip(start, stop)

    # Load face detector
    face_cascade = cv2.CascadeClassifier(f"/root/{FACE_CASCADE_FN}")

    # Run face detector on frames
    imgs = []
    for img in clip.iter_frames():
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        imgs.append(img)

    # Create mp4 of result
    out_clip = moviepy.editor.ImageSequenceClip(imgs, fps=clip.fps)
    out_fn = f"/clips/{start:04d}.mp4"
    out_clip.write_videofile(out_fn)
    return out_fn


# ### Modal entrypoint function
#
# This 'entrypoint' into Modal controls the main flow of the program:
#
# 1. Download the video from YouTube
# 2. Fan-out face detection of individual 1s clips
# 3. Stitch the results back into a new video


@stub.function(shared_volumes={"/clips": stub.sv})
def process_video(url):
    print(f"Downloading video from '{url}'")
    yt = pytube.YouTube(url)
    stream = yt.streams.filter(file_extension="mp4").first()
    fn = stream.download(output_path="/clips/")

    # Get duration
    duration = moviepy.editor.VideoFileClip(fn).duration

    # Create (start, stop) intervals
    intervals = [(fn, offset, offset + 1) for offset in range(int(duration))]

    print("Processing each range of 1s intervals using a Modal map")
    out_fns = list(detect_faces.starmap(intervals))

    print("Converting detections to video clips")
    out_clips = [moviepy.editor.VideoFileClip(out_fn) for out_fn in out_fns]

    print("Concatenating results")
    final_clip = moviepy.editor.concatenate_videoclips(out_clips)
    final_fn = "/clips/out.mp4"
    final_clip.write_videofile(final_fn)

    # Return the full image data
    with open(final_fn, "rb") as f:
        return os.path.basename(fn), f.read()


# ### Local entrypoint
#
# The code we run locally to fire up the Modal job is quite simple
#
# * Take a YouTube URL on the command line
# * Run the Modal function
# * Store the output data


@stub.local_entrypoint
def main(youtube_url: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"):
    fn, movie_data = process_video.call(youtube_url)
    abs_fn = os.path.join(OUTPUT_DIR, fn)
    print(f"writing results to {abs_fn}")
    with open(abs_fn, "wb") as f:
        f.write(movie_data)


# ## Running the script
#
# Running this script should take approximately a minute or less.
# It might output a lot of warnings to standard error.
# These are generally harmless.
#
# Note that we don't preserve the sound in the video.
#
# ## Further directions
#
# As you can tell from the resulting video, this face detection model is not state of the art.
# It has plenty of false positives (non-faces being labeled faces) and false negatives (real faces not being labeled).
# For better model, consider a modern one based on deep learning.
