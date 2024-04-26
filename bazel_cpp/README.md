# Installation (currently only works on linux)

* Install Bazelisk
    * `wget https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64`
    * `chmod +x bazelisk-linux-amd64`
    * `sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel`
* `python3.11 -m venv venv --prompt modal-cpp-example`
* `source venv/bin/activate`
* `pip install modal`

# Running example

* `bazel build //main:hello-world --sandbox_add_mount_pair=/tmp`
* `source venv/bin/activate`
* `modal run run_binary_modal.py`