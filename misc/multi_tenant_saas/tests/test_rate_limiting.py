"""
Tests for rate limiting functionality.
"""

import pytest
from config import get_rate_limit
from rate_limiter import RateLimiter


def test_rate_limit_config():
    """Test rate limit configuration for different tiers."""
    free_config = get_rate_limit("free")
    pro_config = get_rate_limit("pro")
    enterprise_config = get_rate_limit("enterprise")

    # Free tier should have lowest limits
    assert free_config["requests_per_hour"] < pro_config["requests_per_hour"]
    assert pro_config["requests_per_hour"] < enterprise_config["requests_per_hour"]

    # Burst limits should also scale
    assert free_config["burst_limit"] < pro_config["burst_limit"]
    assert pro_config["burst_limit"] < enterprise_config["burst_limit"]


@pytest.mark.asyncio
async def test_rate_limit_basic(mock_rate_limit_dict):
    """Test basic rate limiting functionality."""
    limiter = RateLimiter(mock_rate_limit_dict)

    # First request should be allowed
    allowed, metadata = await limiter.check_rate_limit("test-tenant", "free")

    assert allowed is True
    assert metadata["requests_remaining"] > 0


@pytest.mark.asyncio
async def test_rate_limit_status(mock_rate_limit_dict):
    """Test getting rate limit status."""
    limiter = RateLimiter(mock_rate_limit_dict)

    # Make some requests
    await limiter.check_rate_limit("test-tenant", "free")
    await limiter.check_rate_limit("test-tenant", "free")

    # Get status
    status = await limiter.get_rate_limit_status("test-tenant", "free")

    assert status["requests_used"] == 2
    assert "requests_remaining" in status
    assert "reset_at" in status


@pytest.mark.asyncio
async def test_concurrent_limit(mock_rate_limit_dict):
    """Test concurrent request limiting."""
    limiter = RateLimiter(mock_rate_limit_dict)

    # Increment concurrent counter
    await limiter.increment_concurrent("test-tenant")
    await limiter.increment_concurrent("test-tenant")

    # Decrement
    await limiter.decrement_concurrent("test-tenant")

    # Should not raise error
    await limiter.check_concurrent_limit("test-tenant", "free")


@pytest.mark.asyncio
async def test_reset_rate_limit(mock_rate_limit_dict):
    """Test resetting rate limits."""
    limiter = RateLimiter(mock_rate_limit_dict)

    # Make some requests
    await limiter.check_rate_limit("test-tenant", "free")
    await limiter.check_rate_limit("test-tenant", "free")

    # Reset
    await limiter.reset_rate_limit("test-tenant")

    # Status should show zero usage
    status = await limiter.get_rate_limit_status("test-tenant", "free")
    assert status["requests_used"] == 0


@pytest.mark.asyncio
async def test_different_tiers(mock_rate_limit_dict):
    """Test that different tiers have different limits."""
    limiter = RateLimiter(mock_rate_limit_dict)

    # Get status for different tiers
    free_status = await limiter.get_rate_limit_status("tenant-free", "free")
    pro_status = await limiter.get_rate_limit_status("tenant-pro", "pro")

    assert (
        free_status["requests_per_hour"]
        < pro_status["requests_per_hour"]
    )


# Mock Dict for testing
class MockDict:
    """Mock Modal Dict for testing."""

    def __init__(self):
        self.data = {}

    async def get_async(self, key):
        """Mock async get operation."""
        if key not in self.data:
            raise KeyError(f"Key not found: {key}")
        return self.data[key]

    async def put_async(self, key, value):
        """Mock async put operation."""
        self.data[key] = value

    async def delete_async(self, key):
        """Mock async delete operation."""
        if key in self.data:
            del self.data[key]

    @property
    def get(self):
        """Property to access async get."""
        class Aio:
            def __init__(aio_self, parent):
                aio_self.parent = parent

            async def __call__(aio_self, key):
                return await aio_self.parent.get_async(key)

        return Aio(self)

    @property
    def put(self):
        """Property to access async put."""
        class Aio:
            def __init__(aio_self, parent):
                aio_self.parent = parent

            async def __call__(aio_self, key, value):
                await aio_self.parent.put_async(key, value)

        return Aio(self)

    @property
    def delete(self):
        """Property to access async delete."""
        class Aio:
            def __init__(aio_self, parent):
                aio_self.parent = parent

            async def __call__(aio_self, key):
                await aio_self.parent.delete_async(key)

        return Aio(self)


@pytest.fixture
def mock_rate_limit_dict():
    """Provide a mock Dict for testing."""
    return MockDict()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
