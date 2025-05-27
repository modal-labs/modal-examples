import os
import shutil
from pathlib import Path

comfy_dir = Path(__file__).parent.parent.parent / "comfy"

model_management_path = str(comfy_dir / "model_management.py")
original_model_management_path = str(comfy_dir / "model_management_original.py")
is_patched = os.path.exists(original_model_management_path)


def _apply_cuda_safe_patch():
    """Apply a permanent patch that avoid torch cuda init during snapshots"""

    shutil.copy(model_management_path, original_model_management_path)
    print(
        "[memory_snapshot_helper] ==> Applying CUDA-safe patch for model_management.py"
    )

    with open(model_management_path, "r") as f:
        content = f.read()

    # Find the get_torch_device function and modify the CUDA device access
    # The original line uses: return torch.device(torch.cuda.current_device())
    # We'll replace it with a check if CUDA is available

    # Define the patched content as a constant
    CUDA_SAFE_PATCH = """import os
        if torch.cuda.is_available():
            return torch.device(torch.cuda.current_device())
        else:
            logging.info("[memory_snapshot_helper] CUDA is not available, defaulting to cpu")
            return torch.device('cpu')  # Safe fallback during snapshot"""

    if "return torch.device(torch.cuda.current_device())" in content:
        patched_content = content.replace(
            "return torch.device(torch.cuda.current_device())", CUDA_SAFE_PATCH
        )

        # Save the patched version
        with open(model_management_path, "w") as f:
            f.write(patched_content)

        print("[memory_snapshot_helper] ==> Successfully patched model_management.py")
    else:
        raise Exception(
            "[memory_snapshot_helper] ==> Failed to patch model_management.py"
        )


if not is_patched:
    _apply_cuda_safe_patch()
