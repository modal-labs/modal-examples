"""
Authentication Middleware for Multi-Tenant SaaS
Handles JWT token generation, validation, and tenant context extraction.
"""
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt
from config import (
    API_KEY_PREFIX,
    JWT_ALGORITHM,
    TOKEN_EXPIRY_HOURS,
)


@dataclass
class TenantContext:
    """Tenant information extracted from authentication."""
    tenant_id: str
    tier: str
    name: str
    created_at: datetime
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "tier": self.tier,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class AuthMiddleware:
    """Handles authentication and authorization for multi-tenant requests."""

    def __init__(self, jwt_secret: str):
        """
        Initialize authentication middleware.

        Args:
            jwt_secret: Secret key for JWT signing and validation
        """
        self.jwt_secret = jwt_secret
        self.algorithm = JWT_ALGORITHM

    def generate_token(
        self,
        tenant_id: str,
        tier: str = "free",
        metadata: Optional[Dict[str, Any]] = None,
        expiry_hours: int = TOKEN_EXPIRY_HOURS,
    ) -> str:
        """
        Generate a JWT token for a tenant.

        Args:
            tenant_id: Unique identifier for the tenant
            tier: Subscription tier (free, pro, enterprise)
            metadata: Additional tenant metadata
            expiry_hours: Token validity in hours

        Returns:
            JWT token string
        """
        now = datetime.utcnow()
        payload = {
            "sub": tenant_id,  # Subject (tenant ID)
            "tier": tier,
            "iat": now,  # Issued at
            "exp": now + timedelta(hours=expiry_hours),  # Expiration
            "jti": f"{tenant_id}_{int(now.timestamp())}",  # JWT ID
            "metadata": metadata or {},
        }

        token = jwt.encode(payload, self.jwt_secret, algorithm=self.algorithm)
        return f"{API_KEY_PREFIX}{token}"

    def validate_token(self, token: str) -> TenantContext:
        """
        Validate a JWT token and extract tenant context.

        Args:
            token: JWT token from Authorization header

        Returns:
            TenantContext with validated tenant information

        Raises:
            AuthenticationError: If token is invalid or expired
        """
        # Remove API key prefix if present
        if token.startswith(API_KEY_PREFIX):
            token = token[len(API_KEY_PREFIX):]

        try:
            # Decode and validate token
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.algorithm],
            )

            # Extract tenant information
            tenant_id = payload.get("sub")
            if not tenant_id:
                raise AuthenticationError("Invalid token: missing tenant ID")

            # Create tenant context
            context = TenantContext(
                tenant_id=tenant_id,
                tier=payload.get("tier", "free"),
                name=payload.get("metadata", {}).get("name", "Unknown"),
                created_at=datetime.fromtimestamp(payload.get("iat", time.time())),
                metadata=payload.get("metadata", {}),
            )

            return context

        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")

    def extract_token_from_header(self, authorization_header: Optional[str]) -> str:
        """
        Extract token from Authorization header.

        Args:
            authorization_header: Value of Authorization header

        Returns:
            Extracted token

        Raises:
            AuthenticationError: If header is missing or malformed
        """
        if not authorization_header:
            raise AuthenticationError("Missing Authorization header")

        parts = authorization_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise AuthenticationError(
                "Invalid Authorization header format. Use: Bearer <token>"
            )

        return parts[1]

    async def authenticate_request(
        self,
        authorization_header: Optional[str],
    ) -> TenantContext:
        """
        Authenticate an incoming request.

        Args:
            authorization_header: Authorization header from request

        Returns:
            Validated tenant context

        Raises:
            AuthenticationError: If authentication fails
        """
        token = self.extract_token_from_header(authorization_header)
        context = self.validate_token(token)
        return context


# Modal function to create auth middleware instance
def create_auth_middleware() -> AuthMiddleware:
    """
    Create an authenticated middleware instance using Modal secrets.

    Returns:
        Configured AuthMiddleware instance
    """
    import os

    jwt_secret = os.environ.get("JWT_SECRET_KEY")
    if not jwt_secret:
        raise ValueError("JWT_SECRET_KEY not found in environment")

    return AuthMiddleware(jwt_secret)


# Helper function to verify admin access
def require_admin(tenant_context: TenantContext) -> None:
    """
    Verify that the tenant has admin privileges.

    Args:
        tenant_context: Authenticated tenant context

    Raises:
        AuthenticationError: If tenant is not admin
    """
    if not tenant_context.metadata.get("is_admin", False):
        raise AuthenticationError("Admin access required")


# Helper function to check tenant tier
def require_tier(tenant_context: TenantContext, min_tier: str) -> None:
    """
    Verify that tenant meets minimum tier requirement.

    Args:
        tenant_context: Authenticated tenant context
        min_tier: Minimum required tier (free, pro, enterprise)

    Raises:
        AuthenticationError: If tenant tier is insufficient
    """
    tier_hierarchy = {"free": 0, "pro": 1, "enterprise": 2}

    tenant_tier_level = tier_hierarchy.get(tenant_context.tier, 0)
    required_tier_level = tier_hierarchy.get(min_tier, 0)

    if tenant_tier_level < required_tier_level:
        raise AuthenticationError(
            f"This feature requires {min_tier} tier or higher. "
            f"Current tier: {tenant_context.tier}"
        )


# Example usage in a Modal Function
"""
@app.function(secrets=[modal.Secret.from_name("jwt-secret")])
async def protected_endpoint(authorization: str, data: dict):
    # Create auth middleware
    auth = create_auth_middleware()

    # Authenticate request
    tenant_context = await auth.authenticate_request(authorization)

    # Now you have secure tenant context
    print(f"Request from tenant: {tenant_context.tenant_id}")
    print(f"Tier: {tenant_context.tier}")

    # Process request with tenant isolation
    result = process_for_tenant(tenant_context, data)
    return result
"""
