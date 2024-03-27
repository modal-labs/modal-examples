# # Memory snapshots for imports
#
# This program enables memory snapshotting for imports. We will create
# a memory snapshot for this program after the import sequence is completed.

import modal  # this import in global scope will be included in the snapshot

image = modal.Image.debian_slim().pip_install("torch==2.2.1")
stub = modal.Stub("example-import-torch-memory-snapshot", image=image)

# All imports made globally and inside the `with image.imports()` block will
# be included in the memory snapshot.

with image.imports():
    import torch  # this import inside the container image will also be included

# The program will start from the snapshot from there onwards, skipping all imports.
@stub.function(enable_memory_snapshot=True)
def run():
    print(torch.__version__)


@stub.local_entrypoint()
def main():
    run.remote()
