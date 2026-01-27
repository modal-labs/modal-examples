"""
Tests for authentication middleware.
"""
from datetime import datetime

import pytest
from auth_middleware import AuthenticationError, AuthMiddleware, TenantContext


def test_token_generation():
    """Test JWT token generation."""
    auth = AuthMiddleware("test-secret-key")

    token = auth.generate_token(
        tenant_id="test-tenant",
        tier="pro",
        metadata={"company": "Test Co"},
    )

    assert token.startswith("mt_")
    assert len(token) > 20


def test_token_validation():
    """Test JWT token validation."""
    auth = AuthMiddleware("test-secret-key")

    # Generate token
    token = auth.generate_token(
        tenant_id="test-tenant",
        tier="pro",
        metadata={"name": "Test Tenant"},
    )

    # Validate token
    context = auth.validate_token(token)

    assert isinstance(context, TenantContext)
    assert context.tenant_id == "test-tenant"
    assert context.tier == "pro"
    assert context.name == "Test Tenant"


def test_expired_token():
    """Test that expired tokens are rejected."""
    auth = AuthMiddleware("test-secret-key")

    # Generate token with very short expiry
    token = auth.generate_token(
        tenant_id="test-tenant",
        tier="free",
        expiry_hours=-1,  # Already expired
    )

    # Should raise AuthenticationError
    with pytest.raises(AuthenticationError, match="expired"):
        auth.validate_token(token)


def test_invalid_token():
    """Test that invalid tokens are rejected."""
    auth = AuthMiddleware("test-secret-key")

    # Try to validate garbage token
    with pytest.raises(AuthenticationError, match="Invalid token"):
        auth.validate_token("mt_invalid_garbage_token")


def test_wrong_secret():
    """Test that tokens signed with different secret are rejected."""
    auth1 = AuthMiddleware("secret-1")
    auth2 = AuthMiddleware("secret-2")

    # Generate token with secret-1
    token = auth1.generate_token(tenant_id="test-tenant", tier="free")

    # Try to validate with secret-2
    with pytest.raises(AuthenticationError):
        auth2.validate_token(token)


def test_extract_token_from_header():
    """Test extracting token from Authorization header."""
    auth = AuthMiddleware("test-secret-key")

    # Valid Bearer token
    token = auth.extract_token_from_header("Bearer mt_abc123")
    assert token == "mt_abc123"

    # Missing header
    with pytest.raises(AuthenticationError, match="Missing"):
        auth.extract_token_from_header(None)

    # Invalid format
    with pytest.raises(AuthenticationError, match="Invalid"):
        auth.extract_token_from_header("mt_abc123")  # Missing "Bearer"

    with pytest.raises(AuthenticationError, match="Invalid"):
        auth.extract_token_from_header("Basic mt_abc123")  # Wrong scheme


@pytest.mark.asyncio
async def test_authenticate_request():
    """Test full request authentication flow."""
    auth = AuthMiddleware("test-secret-key")

    # Generate valid token
    token = auth.generate_token(
        tenant_id="test-tenant",
        tier="enterprise",
    )

    # Create Authorization header
    auth_header = f"Bearer {token}"

    # Authenticate request
    context = await auth.authenticate_request(auth_header)

    assert context.tenant_id == "test-tenant"
    assert context.tier == "enterprise"


def test_tenant_context_to_dict():
    """Test TenantContext serialization."""
    context = TenantContext(
        tenant_id="test-tenant",
        tier="pro",
        name="Test Corp",
        created_at=datetime.utcnow(),
        metadata={"company": "Test", "industry": "Tech"},
    )

    data = context.to_dict()

    assert data["tenant_id"] == "test-tenant"
    assert data["tier"] == "pro"
    assert data["name"] == "Test Corp"
    assert "created_at" in data
    assert data["metadata"]["company"] == "Test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
