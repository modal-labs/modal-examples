"""
Configuration for Multi-Tenant SaaS Isolation Engine
"""
from typing import Any, Dict

# Authentication
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24
API_KEY_PREFIX = "mt_"

# Rate Limiting (requests per time window)
RATE_LIMIT_CONFIG = {
    "free": {
        "requests_per_hour": 100,
        "burst_limit": 20,
        "max_concurrent": 5,
    },
    "pro": {
        "requests_per_hour": 1000,
        "burst_limit": 100,
        "max_concurrent": 20,
    },
    "enterprise": {
        "requests_per_hour": 10000,
        "burst_limit": 500,
        "max_concurrent": 100,
    },
}

DEFAULT_TIER = "free"

# Resource Limits per Tenant
RESOURCE_LIMITS = {
    "free": {
        "max_compute_seconds_per_day": 3600,  # 1 hour
        "max_storage_gb": 1,
        "max_volume_files": 1000,
    },
    "pro": {
        "max_compute_seconds_per_day": 36000,  # 10 hours
        "max_storage_gb": 50,
        "max_volume_files": 100000,
    },
    "enterprise": {
        "max_compute_seconds_per_day": 360000,  # 100 hours
        "max_storage_gb": 500,
        "max_volume_files": 1000000,
    },
}

# Volume Configuration
VOLUME_PREFIX = "tenant"
AUTO_CREATE_VOLUMES = True
VOLUME_MOUNT_PATH = "/tenant-data"
SHARED_CACHE_VOLUME = "shared-cache"  # For shared resources like models

# Modal Configuration
MODAL_APP_NAME = "multi-tenant-saas"
MODAL_IMAGE_PACKAGES = [
    "fastapi[standard]",
    "pyjwt",
    "redis",
    "pydantic",
]

# Worker Configuration
DEFAULT_CPU = 1
DEFAULT_MEMORY = 2048  # MB
DEFAULT_TIMEOUT = 300  # seconds
MAX_RETRIES = 3

# Monitoring & Logging
ENABLE_METRICS = True
ENABLE_AUDIT_LOG = True
METRICS_RETENTION_DAYS = 90

# Dict/Storage Keys
TENANT_METADATA_DICT = "tenant-metadata"
RATE_LIMIT_DICT = "rate-limits"
USAGE_STATS_DICT = "usage-stats"
AUDIT_LOG_DICT = "audit-logs"

# Health Check
HEALTH_CHECK_INTERVAL_SECONDS = 60
IDLE_TIMEOUT_MINUTES = 30

# Security
ALLOWED_ORIGINS = [
    "https://yourdomain.com",
    "https://*.yourdomain.com",
]
REQUIRE_HTTPS = True
MAX_REQUEST_SIZE_MB = 10


def get_rate_limit(tier: str = DEFAULT_TIER) -> Dict[str, int]:
    """Get rate limit configuration for a tenant tier."""
    return RATE_LIMIT_CONFIG.get(tier, RATE_LIMIT_CONFIG[DEFAULT_TIER])


def get_resource_limits(tier: str = DEFAULT_TIER) -> Dict[str, Any]:
    """Get resource limits for a tenant tier."""
    return RESOURCE_LIMITS.get(tier, RESOURCE_LIMITS[DEFAULT_TIER])


def get_volume_name(tenant_id: str, volume_type: str = "data") -> str:
    """Generate volume name for a tenant."""
    return f"{VOLUME_PREFIX}-{tenant_id}-{volume_type}"


def get_dict_key(tenant_id: str, key: str) -> str:
    """Generate namespaced dict key for a tenant."""
    return f"{tenant_id}:{key}"
