"""
Rate Limiter - Per-tenant rate limiting to prevent resource abuse.
"""
import json
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import modal
from config import (
    RATE_LIMIT_DICT,
    get_dict_key,
    get_rate_limit,
)


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimiter:
    """
    Per-tenant rate limiter using sliding window algorithm.
    Prevents a single tenant from consuming excessive resources.
    """

    def __init__(self, rate_limit_dict: modal.Dict):
        """
        Initialize rate limiter.

        Args:
            rate_limit_dict: Modal Dict for storing rate limit counters
        """
        self.rate_limit_dict = rate_limit_dict

    async def check_rate_limit(
        self,
        tenant_id: str,
        tier: str = "free",
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Check if request should be allowed based on rate limits.

        Args:
            tenant_id: Tenant identifier
            tier: Tenant subscription tier

        Returns:
            Tuple of (allowed: bool, metadata: dict)

        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        config = get_rate_limit(tier)
        requests_per_hour = config["requests_per_hour"]
        burst_limit = config["burst_limit"]

        now = time.time()
        window_start = now - 3600  # 1 hour window

        # Get current rate limit data
        key = get_dict_key(tenant_id, f"rate_limit:{int(now // 60)}")  # Per-minute key

        try:
            data = await self.rate_limit_dict.get.aio(key)
            rate_data = json.loads(data)
        except KeyError:
            # No data exists, create new
            rate_data = {
                "requests": [],
                "hourly_count": 0,
                "window_start": window_start,
            }

        # Clean old requests outside the window
        rate_data["requests"] = [
            req_time for req_time in rate_data["requests"]
            if req_time > window_start
        ]

        # Check burst limit (requests in last minute)
        recent_requests = [
            req_time for req_time in rate_data["requests"]
            if req_time > now - 60
        ]

        if len(recent_requests) >= burst_limit:
            retry_after = 60 - int(now - min(recent_requests))
            raise RateLimitExceeded(
                f"Burst limit exceeded. Maximum {burst_limit} requests per minute.",
                retry_after=retry_after,
            )

        # Check hourly limit
        if len(rate_data["requests"]) >= requests_per_hour:
            retry_after = int(3600 - (now - min(rate_data["requests"])))
            raise RateLimitExceeded(
                f"Rate limit exceeded. Maximum {requests_per_hour} requests per hour.",
                retry_after=retry_after,
            )

        # Allow request - add to counter
        rate_data["requests"].append(now)
        rate_data["hourly_count"] = len(rate_data["requests"])

        # Save updated rate data
        await self.rate_limit_dict.put.aio(key, json.dumps(rate_data))

        # Return metadata about current usage
        metadata = {
            "requests_used": len(rate_data["requests"]),
            "requests_remaining": requests_per_hour - len(rate_data["requests"]),
            "reset_at": datetime.fromtimestamp(
                min(rate_data["requests"]) + 3600
            ).isoformat(),
        }

        return True, metadata

    async def get_rate_limit_status(
        self,
        tenant_id: str,
        tier: str = "free",
    ) -> Dict[str, any]:
        """
        Get current rate limit status for a tenant.

        Args:
            tenant_id: Tenant identifier
            tier: Tenant subscription tier

        Returns:
            Current rate limit status
        """
        config = get_rate_limit(tier)
        now = time.time()
        window_start = now - 3600

        key = get_dict_key(tenant_id, f"rate_limit:{int(now // 60)}")

        try:
            data = await self.rate_limit_dict.get.aio(key)
            rate_data = json.loads(data)
        except KeyError:
            return {
                "requests_used": 0,
                "requests_remaining": config["requests_per_hour"],
                "requests_per_hour": config["requests_per_hour"],
                "burst_limit": config["burst_limit"],
            }

        # Clean old requests
        rate_data["requests"] = [
            req_time for req_time in rate_data["requests"]
            if req_time > window_start
        ]

        return {
            "requests_used": len(rate_data["requests"]),
            "requests_remaining": config["requests_per_hour"] - len(rate_data["requests"]),
            "requests_per_hour": config["requests_per_hour"],
            "burst_limit": config["burst_limit"],
            "reset_at": datetime.fromtimestamp(
                min(rate_data["requests"]) + 3600 if rate_data["requests"] else now
            ).isoformat(),
        }

    async def reset_rate_limit(self, tenant_id: str) -> None:
        """
        Reset rate limit for a tenant (admin operation).

        Args:
            tenant_id: Tenant identifier
        """
        now = time.time()
        key = get_dict_key(tenant_id, f"rate_limit:{int(now // 60)}")

        try:
            await self.rate_limit_dict.delete.aio(key)
            print(f"Rate limit reset for tenant {tenant_id}")
        except KeyError:
            pass  # No rate limit data exists

    async def check_concurrent_limit(
        self,
        tenant_id: str,
        tier: str = "free",
    ) -> bool:
        """
        Check if tenant has exceeded concurrent request limit.

        Args:
            tenant_id: Tenant identifier
            tier: Tenant subscription tier

        Returns:
            True if within limit, False otherwise
        """
        config = get_rate_limit(tier)
        max_concurrent = config["max_concurrent"]

        key = get_dict_key(tenant_id, "concurrent_requests")

        try:
            data = await self.rate_limit_dict.get.aio(key)
            concurrent_data = json.loads(data)
            current_count = concurrent_data.get("count", 0)
        except KeyError:
            current_count = 0

        if current_count >= max_concurrent:
            raise RateLimitExceeded(
                f"Concurrent request limit exceeded. Maximum {max_concurrent} concurrent requests.",
                retry_after=5,
            )

        return True

    async def increment_concurrent(self, tenant_id: str) -> None:
        """
        Increment concurrent request counter.

        Args:
            tenant_id: Tenant identifier
        """
        key = get_dict_key(tenant_id, "concurrent_requests")

        try:
            data = await self.rate_limit_dict.get.aio(key)
            concurrent_data = json.loads(data)
        except KeyError:
            concurrent_data = {"count": 0, "updated_at": time.time()}

        concurrent_data["count"] += 1
        concurrent_data["updated_at"] = time.time()

        await self.rate_limit_dict.put.aio(key, json.dumps(concurrent_data))

    async def decrement_concurrent(self, tenant_id: str) -> None:
        """
        Decrement concurrent request counter.

        Args:
            tenant_id: Tenant identifier
        """
        key = get_dict_key(tenant_id, "concurrent_requests")

        try:
            data = await self.rate_limit_dict.get.aio(key)
            concurrent_data = json.loads(data)
            concurrent_data["count"] = max(0, concurrent_data["count"] - 1)
            concurrent_data["updated_at"] = time.time()

            await self.rate_limit_dict.put.aio(key, json.dumps(concurrent_data))
        except KeyError:
            pass  # No data exists


def create_rate_limiter() -> RateLimiter:
    """
    Create a rate limiter instance with Modal resources.

    Returns:
        Configured RateLimiter instance
    """
    rate_limit_dict = modal.Dict.from_name(
        RATE_LIMIT_DICT,
        create_if_missing=True,
    )

    return RateLimiter(rate_limit_dict)


# Decorator for easy rate limiting
def rate_limited(tier_override: Optional[str] = None):
    """
    Decorator to add rate limiting to a Modal function.

    Args:
        tier_override: Override tier (useful for testing)

    Usage:
        @app.function()
        @rate_limited()
        async def my_function(tenant_id: str, tier: str, data: dict):
            # Function is automatically rate limited
            pass
    """
    def decorator(func):
        async def wrapper(tenant_id: str, tier: str, *args, **kwargs):
            limiter = create_rate_limiter()

            # Check rate limit
            actual_tier = tier_override or tier
            try:
                allowed, metadata = await limiter.check_rate_limit(tenant_id, actual_tier)

                if not allowed:
                    raise RateLimitExceeded(
                        "Rate limit exceeded",
                        retry_after=metadata.get("retry_after", 60),
                    )

                # Track concurrent requests
                await limiter.check_concurrent_limit(tenant_id, actual_tier)
                await limiter.increment_concurrent(tenant_id)

                try:
                    # Execute function
                    result = await func(tenant_id, tier, *args, **kwargs)
                    return result
                finally:
                    # Always decrement concurrent counter
                    await limiter.decrement_concurrent(tenant_id)

            except RateLimitExceeded:
                raise  # Re-raise rate limit exceptions

        return wrapper
    return decorator


# Example usage
"""
@app.function()
async def check_limits_example(tenant_id: str):
    limiter = create_rate_limiter()

    # Check if request is allowed
    try:
        allowed, metadata = await limiter.check_rate_limit(tenant_id, "pro")
        print(f"Request allowed. Remaining: {metadata['requests_remaining']}")
    except RateLimitExceeded as e:
        print(f"Rate limit exceeded: {e}")
        print(f"Retry after: {e.retry_after} seconds")

    # Get current status
    status = await limiter.get_rate_limit_status(tenant_id, "pro")
    print(f"Current usage: {status}")
"""
