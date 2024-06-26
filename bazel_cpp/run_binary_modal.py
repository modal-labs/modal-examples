from modal import App, Mount
import os

app = App("example-modal-bazel-cpp")

CWD = os.getcwd()
LOCAL_REL_PATH= "bazel-bin/main/hello-world"
REMOTE_PATH = "/root/my-binary"

@app.function(mounts=[Mount.from_local_file(local_path=f"{CWD}/{LOCAL_REL_PATH}", remote_path=REMOTE_PATH)])
def f():
    import subprocess
    subprocess.run(REMOTE_PATH)