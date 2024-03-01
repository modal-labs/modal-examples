# # Memory snapshots for imports
#
# This program enables memory snapshotting for imports. We will create
# a memory snapshot for this program after the import sequence is completed.
# The program will start from the snapshot from there onwards.

import modal

image = modal.Image.debian_slim().pip_install("torch==2.2.1")
stub = modal.Stub("import-torch", image=image)

# All imports made up to this point will be included in the snapshot.

with image.imports():
    import torch

@stub.function(checkpointing_enabled=True)
def run():
    print(torch.__version__)


@stub.local_entrypoint()
def main():
    run.remote()
