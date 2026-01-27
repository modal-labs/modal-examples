"""
Worker Functions - Example business logic with tenant isolation.
"""
import time
from typing import Any, Dict

import modal
from config import (
    DEFAULT_CPU,
    DEFAULT_MEMORY,
    DEFAULT_TIMEOUT,
    MAX_RETRIES,
    MODAL_APP_NAME,
    MODAL_IMAGE_PACKAGES,
)
from volume_manager import create_volume_manager

# Create Modal image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*MODAL_IMAGE_PACKAGES)
)

# Create Modal app
app = modal.App(MODAL_APP_NAME, image=image)


@app.function(
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY,
    timeout=DEFAULT_TIMEOUT,
    retries=MAX_RETRIES,
)
def example_worker(
    tenant_id: str,
    task_type: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Example worker function with tenant isolation.

    This function demonstrates how to:
    1. Access tenant-scoped volumes
    2. Process tenant data in isolation
    3. Return results with proper tracking

    Args:
        tenant_id: Tenant identifier
        task_type: Type of task to perform
        data: Task data

    Returns:
        Task results
    """
    start_time = time.time()

    # Create volume manager for this tenant
    vm = create_volume_manager(tenant_id)

    # Mount volumes for this tenant
    vm.get_mount_config(["data", "cache"])

    # Dynamically attach volumes (Note: In real usage, volumes should be
    # declared in the function decorator. This is just for illustration.)

    try:
        # Save input data to tenant volume
        vm.write_file(
            f"task_{int(start_time)}_input.json",
            data,
            volume_type="data",
            commit=False,
        )

        # Process based on task type
        if task_type == "analyze":
            result = _analyze_data(tenant_id, data, vm)
        elif task_type == "transform":
            result = _transform_data(tenant_id, data, vm)
        elif task_type == "aggregate":
            result = _aggregate_data(tenant_id, data, vm)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # Save results to tenant volume
        vm.write_file(
            f"task_{int(start_time)}_output.json",
            result,
            volume_type="data",
            commit=True,  # Persist results
        )

        elapsed = time.time() - start_time

        return {
            "success": True,
            "tenant_id": tenant_id,
            "task_type": task_type,
            "result": result,
            "elapsed_seconds": elapsed,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "tenant_id": tenant_id,
            "task_type": task_type,
            "error": str(e),
            "elapsed_seconds": elapsed,
        }


def _analyze_data(
    tenant_id: str,
    data: Dict[str, Any],
    vm: Any,
) -> Dict[str, Any]:
    """
    Example analysis logic.

    Args:
        tenant_id: Tenant identifier
        data: Input data
        vm: Volume manager

    Returns:
        Analysis results
    """
    # Simulate analysis
    items = data.get("items", [])

    analysis = {
        "total_items": len(items),
        "unique_values": len(set(items)) if items else 0,
        "tenant_id": tenant_id,
    }

    # Could access previous results from volume
    try:
        previous_files = vm.list_files(volume_type="data", pattern="*output.json")
        analysis["previous_analyses"] = len(previous_files)
    except Exception:
        analysis["previous_analyses"] = 0

    return analysis


def _transform_data(
    tenant_id: str,
    data: Dict[str, Any],
    vm: Any,
) -> Dict[str, Any]:
    """
    Example transformation logic.

    Args:
        tenant_id: Tenant identifier
        data: Input data
        vm: Volume manager

    Returns:
        Transformed data
    """
    # Simulate transformation
    items = data.get("items", [])
    operation = data.get("operation", "uppercase")

    if operation == "uppercase" and all(isinstance(x, str) for x in items):
        transformed = [x.upper() for x in items]
    elif operation == "double" and all(isinstance(x, (int, float)) for x in items):
        transformed = [x * 2 for x in items]
    else:
        transformed = items

    return {
        "transformed_items": transformed,
        "operation": operation,
        "count": len(transformed),
    }


def _aggregate_data(
    tenant_id: str,
    data: Dict[str, Any],
    vm: Any,
) -> Dict[str, Any]:
    """
    Example aggregation logic accessing multiple files.

    Args:
        tenant_id: Tenant identifier
        data: Input data
        vm: Volume manager

    Returns:
        Aggregated results
    """
    # Get all previous output files
    output_files = vm.list_files(volume_type="data", pattern="*output.json")

    # Aggregate data from previous tasks
    all_results = []
    for filename in output_files[-10:]:  # Last 10 results
        try:
            result = vm.read_file(filename, volume_type="data", as_json=True)
            all_results.append(result)
        except Exception:
            pass

    return {
        "aggregated_count": len(all_results),
        "tenant_id": tenant_id,
        "sample_results": all_results[:3],  # Return sample
    }


@app.function(
    cpu=2,
    memory=4096,
    timeout=600,
)
def batch_worker(
    tenant_id: str,
    tasks: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """
    Process multiple tasks in batch for a tenant.

    Args:
        tenant_id: Tenant identifier
        tasks: List of tasks to process

    Returns:
        List of results
    """
    results = []

    for task in tasks:
        result = example_worker.local(
            tenant_id=tenant_id,
            task_type=task.get("task_type", "analyze"),
            data=task.get("data", {}),
        )
        results.append(result)

    return results


@app.function(
    gpu="any",  # Example GPU function
    timeout=1800,
)
def ml_inference_worker(
    tenant_id: str,
    model_name: str,
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Example ML inference with GPU and tenant isolation.

    Args:
        tenant_id: Tenant identifier
        model_name: Name of model to use
        inputs: Model inputs

    Returns:
        Inference results
    """
    start_time = time.time()

    # Create volume manager
    vm = create_volume_manager(tenant_id)

    # In a real implementation, you would:
    # 1. Load model from shared cache or tenant-specific weights
    # 2. Run inference
    # 3. Save results to tenant volume

    # Simulate inference
    time.sleep(0.5)  # Simulate GPU computation

    result = {
        "model": model_name,
        "predictions": ["class_a", "class_b"],  # Dummy predictions
        "confidence": [0.85, 0.15],
        "inference_time": time.time() - start_time,
    }

    # Save results
    vm.write_file(
        f"inference_{int(start_time)}.json",
        result,
        volume_type="cache",
        commit=True,
    )

    return result


@app.function(
    schedule=modal.Period(hours=1),  # Run every hour
)
def cleanup_worker():
    """
    Periodic cleanup task for all tenants.

    This demonstrates how to run maintenance tasks.
    In production, you'd iterate through all tenants.
    """
    print("Running cleanup for old files...")

    # In production, you would:
    # 1. Get list of all tenants
    # 2. For each tenant, cleanup old files
    # 3. Record cleanup metrics

    # Example for one tenant
    tenant_id = "example-tenant"
    vm = create_volume_manager(tenant_id)

    deleted = vm.cleanup_old_files(
        volume_type="cache",
        days_old=7,
        commit=True,
    )

    print(f"Deleted {deleted} old files for {tenant_id}")


# Export functions for use in dispatcher
__all__ = [
    "example_worker",
    "batch_worker",
    "ml_inference_worker",
    "cleanup_worker",
]
