"""
Dispatcher - Main API Gateway for Multi-Tenant SaaS.

This is the entry point for all tenant requests. It handles:
1. Authentication & authorization
2. Rate limiting
3. Tenant isolation
4. Request routing to appropriate workers
5. Usage tracking
"""
import time
from typing import Any, Dict, Optional

import modal
from auth_middleware import (
    AuthenticationError,
    TenantContext,
    create_auth_middleware,
)
from config import (
    ALLOWED_ORIGINS,
    MODAL_APP_NAME,
    MODAL_IMAGE_PACKAGES,
)
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rate_limiter import RateLimitExceeded, create_rate_limiter
from tenant_manager import create_tenant_manager
from usage_tracker import create_usage_tracker
from worker_functions import batch_worker, example_worker, ml_inference_worker

# Create Modal image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*MODAL_IMAGE_PACKAGES)
)

# Create Modal app
app = modal.App(MODAL_APP_NAME, image=image)

# Create FastAPI app
web_app = FastAPI(
    title="Multi-Tenant SaaS API",
    description="Production-ready multi-tenant API with isolation",
    version="1.0.0",
)

# Add CORS middleware
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency for authentication
async def get_tenant_context(
    authorization: Optional[str] = Header(None),
) -> TenantContext:
    """
    Extract and validate tenant context from request.

    Args:
        authorization: Authorization header

    Returns:
        Validated tenant context

    Raises:
        HTTPException: If authentication fails
    """
    try:
        auth = create_auth_middleware()
        context = await auth.authenticate_request(authorization)
        return context
    except AuthenticationError as e:
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


# Middleware for rate limiting and usage tracking
@web_app.middleware("http")
async def rate_limit_and_track(request: Request, call_next):
    """
    Middleware to enforce rate limits and track usage.
    """
    # Skip for health check and docs
    if request.url.path in ["/health", "/docs", "/openapi.json"]:
        return await call_next(request)

    start_time = time.time()

    try:
        # Get tenant context from request
        auth_header = request.headers.get("authorization")
        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={"error": "Missing authorization header"},
            )

        auth = create_auth_middleware()
        tenant_context = await auth.authenticate_request(auth_header)

        # Check rate limits
        rate_limiter = create_rate_limiter()
        try:
            allowed, metadata = await rate_limiter.check_rate_limit(
                tenant_context.tenant_id,
                tenant_context.tier,
            )

            # Add rate limit info to response headers
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(metadata["requests_used"] + metadata["requests_remaining"])
            response.headers["X-RateLimit-Remaining"] = str(metadata["requests_remaining"])
            response.headers["X-RateLimit-Reset"] = metadata["reset_at"]

        except RateLimitExceeded as e:
            return JSONResponse(
                status_code=429,
                content={
                    "error": str(e),
                    "retry_after": e.retry_after,
                },
                headers={"Retry-After": str(e.retry_after)},
            )

        # Track usage
        elapsed = time.time() - start_time
        tracker = create_usage_tracker()
        await tracker.record_request(
            tenant_id=tenant_context.tenant_id,
            compute_seconds=elapsed,
            error=response.status_code >= 400,
        )

        return response

    except AuthenticationError as e:
        return JSONResponse(
            status_code=401,
            content={"error": str(e)},
        )
    except Exception as e:
        print(f"Middleware error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )


# Health check endpoint
@web_app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
    }


# Process endpoint - main worker dispatch
@web_app.post("/process")
async def process_task(
    request_data: Dict[str, Any],
    tenant_context: TenantContext = Depends(get_tenant_context),
):
    """
    Process a task for a tenant.

    Args:
        request_data: Task data including task_type and data
        tenant_context: Authenticated tenant context

    Returns:
        Task results
    """
    task_type = request_data.get("task_type", "analyze")
    data = request_data.get("data", {})

    try:
        # Call worker function with tenant isolation
        result = example_worker.remote(
            tenant_id=tenant_context.tenant_id,
            task_type=task_type,
            data=data,
        )

        return {
            "success": True,
            "tenant_id": tenant_context.tenant_id,
            "result": result,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Task processing failed: {str(e)}",
        )


# Batch processing endpoint
@web_app.post("/batch")
async def process_batch(
    request_data: Dict[str, Any],
    tenant_context: TenantContext = Depends(get_tenant_context),
):
    """
    Process multiple tasks in batch.

    Args:
        request_data: Batch data including tasks list
        tenant_context: Authenticated tenant context

    Returns:
        Batch results
    """
    tasks = request_data.get("tasks", [])

    if not tasks:
        raise HTTPException(status_code=400, detail="No tasks provided")

    if len(tasks) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 tasks per batch")

    try:
        results = batch_worker.remote(
            tenant_id=tenant_context.tenant_id,
            tasks=tasks,
        )

        return {
            "success": True,
            "tenant_id": tenant_context.tenant_id,
            "batch_size": len(tasks),
            "results": results,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}",
        )


# ML inference endpoint
@web_app.post("/inference")
async def ml_inference(
    request_data: Dict[str, Any],
    tenant_context: TenantContext = Depends(get_tenant_context),
):
    """
    Run ML model inference.

    Args:
        request_data: Inference request with model_name and inputs
        tenant_context: Authenticated tenant context

    Returns:
        Inference results
    """
    model_name = request_data.get("model_name")
    inputs = request_data.get("inputs", {})

    if not model_name:
        raise HTTPException(status_code=400, detail="model_name required")

    try:
        result = ml_inference_worker.remote(
            tenant_id=tenant_context.tenant_id,
            model_name=model_name,
            inputs=inputs,
        )

        return {
            "success": True,
            "tenant_id": tenant_context.tenant_id,
            "result": result,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )


# Usage stats endpoint
@web_app.get("/usage")
async def get_usage_stats(
    tenant_context: TenantContext = Depends(get_tenant_context),
    date: Optional[str] = None,
):
    """
    Get usage statistics for tenant.

    Args:
        tenant_context: Authenticated tenant context
        date: Optional date (ISO format)

    Returns:
        Usage statistics
    """
    try:
        tracker = create_usage_tracker()
        usage = await tracker.get_usage(tenant_context.tenant_id, date)

        return {
            "tenant_id": tenant_context.tenant_id,
            "date": usage.date,
            "usage": usage.to_dict(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve usage: {str(e)}",
        )


# Rate limit status endpoint
@web_app.get("/rate-limit")
async def get_rate_limit_status(
    tenant_context: TenantContext = Depends(get_tenant_context),
):
    """
    Get current rate limit status.

    Args:
        tenant_context: Authenticated tenant context

    Returns:
        Rate limit status
    """
    try:
        rate_limiter = create_rate_limiter()
        status = await rate_limiter.get_rate_limit_status(
            tenant_context.tenant_id,
            tenant_context.tier,
        )

        return {
            "tenant_id": tenant_context.tenant_id,
            "tier": tenant_context.tier,
            "rate_limit": status,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve rate limit status: {str(e)}",
        )


# Tenant info endpoint
@web_app.get("/tenant")
async def get_tenant_info(
    tenant_context: TenantContext = Depends(get_tenant_context),
):
    """
    Get tenant information.

    Args:
        tenant_context: Authenticated tenant context

    Returns:
        Tenant information
    """
    try:
        tenant_manager = create_tenant_manager()
        tenant_info = await tenant_manager.get_tenant(tenant_context.tenant_id)

        return {
            "tenant": tenant_info,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve tenant info: {str(e)}",
        )


# Mount FastAPI app to Modal
@app.function(
    secrets=[modal.Secret.from_name("jwt-secret")],
    keep_warm=1,  # Keep at least one instance warm
    container_idle_timeout=300,
)
@modal.asgi_app()
def dispatcher():
    """
    Main dispatcher function - serves the FastAPI app.
    """
    return web_app


# Local entrypoint for testing
@app.local_entrypoint()
def main():
    """
    Local entrypoint for development and testing.
    """
    print("Multi-Tenant SaaS Dispatcher")
    print("=" * 50)
    print("\nEndpoints:")
    print("  POST /process      - Process single task")
    print("  POST /batch        - Process batch tasks")
    print("  POST /inference    - ML model inference")
    print("  GET  /usage        - Get usage statistics")
    print("  GET  /rate-limit   - Get rate limit status")
    print("  GET  /tenant       - Get tenant info")
    print("  GET  /health       - Health check")
    print("\nDeploy with:")
    print("  modal deploy dispatcher.py")
    print("\nOr run locally:")
    print("  modal serve dispatcher.py")
