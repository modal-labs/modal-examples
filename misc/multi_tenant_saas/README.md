# Multi-Tenant SaaS Isolation Engine for Modal

A production-ready multi-tenant isolation system for Modal that enables secure, scalable SaaS applications with proper tenant isolation, authentication, and resource management.

## Overview

This example demonstrates how to build a production SaaS backend on Modal with:

- **Tenant Isolation**: Each customer gets isolated compute, storage, and data
- **Dynamic Provisioning**: Volumes and resources are mounted per-tenant on-demand
- **Authentication & Authorization**: JWT-based auth with tenant context
- **Rate Limiting**: Per-tenant rate limits to prevent resource abuse
- **Usage Tracking**: Monitor and bill based on tenant resource consumption
- **Volume Namespacing**: Automatic tenant-scoped volume management

## Architecture

```
┌─────────────────┐
│   API Gateway   │ ← Public endpoint with auth
│   (Dispatcher)  │
└────────┬────────┘
         │
         ├─→ Auth Middleware (JWT validation)
         ├─→ Rate Limiter (per-tenant)
         ├─→ Tenant Manager (isolation)
         │
         ▼
┌─────────────────────────────┐
│  Dynamic Worker Containers  │
│  (Tenant-scoped volumes)    │
└─────────────────────────────┘
```

## Features

### 1. Multi-Tenant Authentication
- JWT-based authentication with tenant context
- API key management per tenant
- Secure token generation and validation

### 2. Tenant Isolation
- Separate Modal Volumes per tenant
- Isolated Dict storage per tenant
- Dynamic volume mounting based on tenant ID

### 3. Rate Limiting
- Per-tenant request limits (prevents one customer from consuming all resources)
- Configurable rate windows and burst limits
- Redis-backed distributed rate limiting

### 4. Usage Tracking
- Track compute time per tenant
- Monitor storage usage per tenant
- Export metrics for billing systems

### 5. Dynamic Provisioning
- Volumes created on-demand for new tenants
- Automatic cleanup of idle resources
- Health checks and monitoring

## Setup

### 1. Install Dependencies

```bash
modal token new  # Authenticate with Modal
pip install -r requirements.txt
```

### 2. Create Modal Secrets

```bash
# Create JWT secret for authentication
modal secret create jwt-secret \
  JWT_SECRET_KEY="your-super-secret-key-change-this" \
  JWT_ALGORITHM="HS256"

# Create Redis credentials (optional, for distributed rate limiting)
modal secret create redis-credentials \
  REDIS_HOST="your-redis-host" \
  REDIS_PORT="6379" \
  REDIS_PASSWORD="your-redis-password"
```

### 3. Deploy the Application

```bash
# Deploy the entire multi-tenant system
modal deploy dispatcher.py

# Or run locally for testing
modal run dispatcher.py
```

## Usage

### 1. Create a Tenant API Key

```bash
# Generate an API key for a new tenant
modal run create_tenant.py --tenant-id "customer-123" --name "Acme Corp"
```

This will output:
```
Created tenant: customer-123
API Key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### 2. Make Authenticated Requests

```python
import requests

# Your tenant's API key
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# Call the API
response = requests.post(
    "https://your-workspace--multi-tenant-dispatcher.modal.run/process",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"task": "analyze", "data": "your-data-here"}
)

print(response.json())
```

### 3. Monitor Usage

```python
# Check tenant usage stats
response = requests.get(
    "https://your-workspace--multi-tenant-usage.modal.run/stats/customer-123",
    headers={"Authorization": f"Bearer {admin_key}"}
)

print(response.json())
# Output:
# {
#   "tenant_id": "customer-123",
#   "requests_today": 1250,
#   "compute_seconds": 3600,
#   "storage_gb": 5.2
# }
```

## File Structure

```
multi_tenant_saas/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── dispatcher.py                      # Main API gateway with routing
├── auth_middleware.py                 # JWT authentication & validation
├── tenant_manager.py                  # Tenant isolation & provisioning
├── rate_limiter.py                    # Per-tenant rate limiting
├── usage_tracker.py                   # Usage metrics & billing
├── volume_manager.py                  # Tenant-scoped volume operations
├── worker_functions.py                # Actual business logic workers
├── create_tenant.py                   # CLI tool to create tenants
├── config.py                          # Configuration constants
└── tests/
    ├── test_auth.py                   # Auth tests
    ├── test_isolation.py              # Isolation tests
    └── test_rate_limiting.py          # Rate limit tests
```

## Configuration

Edit `config.py` to customize:

```python
# Rate limiting
DEFAULT_RATE_LIMIT = 1000  # requests per hour per tenant
BURST_LIMIT = 100          # max burst requests

# Resource limits
MAX_COMPUTE_MINUTES_PER_DAY = 1000
MAX_STORAGE_GB = 50

# Volume settings
VOLUME_PREFIX = "tenant"
AUTO_CREATE_VOLUMES = True
```

## Security Considerations

1. **JWT Secrets**: Use strong, randomly generated secrets
2. **API Keys**: Rotate regularly and store securely
3. **Volume Isolation**: Never mix tenant data in shared volumes
4. **Rate Limiting**: Set conservative limits to prevent abuse
5. **Input Validation**: Always validate tenant inputs
6. **Audit Logging**: Log all tenant operations for compliance

## Advanced Features

### Custom Worker Logic

Add your business logic in `worker_functions.py`:

```python
@app.function(volumes={"/tenant-data": tenant_volume})
def custom_processing(tenant_id: str, data: dict):
    # Your custom processing logic here
    # Access tenant-specific data from /tenant-data
    with open(f"/tenant-data/{tenant_id}/config.json") as f:
        config = json.load(f)
    
    # Process data...
    result = process(data, config)
    
    # Save results to tenant volume
    with open(f"/tenant-data/{tenant_id}/results.json", "w") as f:
        json.dump(result, f)
    
    tenant_volume.commit()
    return result
```

### Integration with Billing Systems

```python
# Export usage data to your billing system
from usage_tracker import get_tenant_usage

usage = get_tenant_usage("customer-123", month="2026-01")
send_to_billing_system(usage)
```

## Testing

```bash
# Run all tests
pytest tests/

# Test authentication
pytest tests/test_auth.py

# Test tenant isolation
pytest tests/test_isolation.py

# Load test rate limiting
pytest tests/test_rate_limiting.py -v
```

## Production Checklist

- [ ] Configure proper JWT secrets
- [ ] Set up monitoring and alerting
- [ ] Configure rate limits based on your SLA
- [ ] Set up backup strategy for tenant volumes
- [ ] Implement audit logging
- [ ] Configure CORS for your frontend domains
- [ ] Set up CI/CD for deployments
- [ ] Create tenant onboarding workflow
- [ ] Implement tenant offboarding and data deletion
- [ ] Set up cost monitoring per tenant

## Troubleshooting

### "Volume not found" errors
- Ensure volumes are created before use
- Check tenant ID is correct in request
- Verify volume names follow pattern: `tenant-{tenant_id}-data`

### Rate limit errors
- Check tenant's current usage
- Verify rate limit configuration
- Consider upgrading tenant's plan

### Authentication failures
- Verify JWT secret is correctly configured
- Check token expiration
- Ensure Authorization header format: `Bearer {token}`

## License

MIT License - Feel free to use in your SaaS projects!

## Contributing

This example is part of the Modal examples repository. Submit PRs to improve multi-tenant patterns!