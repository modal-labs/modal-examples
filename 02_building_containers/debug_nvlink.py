# Debug script to see nvidia-smi nvlink output

import subprocess

import modal

nvshmem_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
).entrypoint([])

app = modal.App("debug-nvlink")


@app.function(gpu="H100:4", image=nvshmem_image, timeout=300)
def debug_nvlink_commands():
    """Test various nvidia-smi nvlink commands to see what output we get."""

    commands = [
        ["nvidia-smi", "nvlink", "-gt", "d"],
        ["nvidia-smi", "nvlink", "-gt", "r"],
        ["nvidia-smi", "nvlink", "--id", "0", "-gt", "d"],
        ["nvidia-smi", "nvlink", "-i", "0", "-gt", "d"],
    ]

    for cmd in commands:
        print("=" * 80)
        print(f"Command: {' '.join(cmd)}")
        print("=" * 80)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10, check=False
            )

            print(f"Return code: {result.returncode}")
            print("\nSTDOUT:")
            print(result.stdout)

            if result.stderr:
                print("\nSTDERR:")
                print(result.stderr)

        except subprocess.TimeoutExpired:
            print("Command timed out")
        except Exception as e:
            print(f"Error: {e}")

        print("\n")


@app.local_entrypoint()
def main():
    debug_nvlink_commands.remote()
