# Quick Start Guide - Multi-Tenant SaaS on Modal

Get your multi-tenant SaaS up and running in 5 minutes!

## Prerequisites

- Modal account ([sign up at modal.com](https://modal.com))
- Python 3.11+
- Modal CLI installed

## Step 1: Install and Authenticate

```bash
# Install Modal
pip install modal

# Authenticate with Modal
modal token new
```

## Step 2: Clone and Setup

```bash
# Navigate to the multi_tenant_saas directory
cd multi_tenant_saas

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Create Secrets

Create your JWT secret for authentication:

```bash
# Generate a secure random secret
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Create the secret in Modal
modal secret create jwt-secret \
  JWT_SECRET_KEY="<paste-the-secret-here>" \
  JWT_ALGORITHM="HS256"
```

## Step 4: Deploy the System

```bash
# Deploy the main API gateway
modal deploy dispatcher.py

# This will output a URL like:
# https://your-workspace--multi-tenant-saas-dispatcher.modal.run
```

## Step 5: Create Your First Tenant

```bash
# Create a test tenant
modal run create_tenant.py create \
  --tenant-id "acme-corp" \
  --name "Acme Corporation" \
  --tier "pro" \
  --email "admin@acme.com"

# This will output an API key - SAVE IT!
```

## Step 6: Test Your API

```bash
# Save your API key and URL
export API_KEY="mt_eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
export API_URL="https://your-workspace--multi-tenant-saas-dispatcher.modal.run"

# Test the health endpoint
curl $API_URL/health

# Test with authentication
curl -X POST $API_URL/process \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "analyze",
    "data": {"items": [1, 2, 3, 4, 5]}
  }'
```

## Step 7: Run Examples

```bash
# Update example_usage.py with your API key and URL
nano example_usage.py

# Run the examples
python example_usage.py
```

## What You Get

✅ **Secure Authentication** - JWT-based auth with tenant isolation  
✅ **Rate Limiting** - Per-tenant rate limits (100/hour free, 1000/hour pro)  
✅ **Usage Tracking** - Monitor compute time, requests, storage  
✅ **Isolated Storage** - Separate Modal Volumes per tenant  
✅ **Auto-Scaling** - Modal handles all scaling automatically  
✅ **Production Ready** - Error handling, retries, monitoring  

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/process` | POST | Process single task |
| `/batch` | POST | Process multiple tasks |
| `/inference` | POST | ML model inference |
| `/usage` | GET | Get usage statistics |
| `/rate-limit` | GET | Check rate limits |
| `/tenant` | GET | Get tenant info |

## Common Tasks

### Create More Tenants

```bash
modal run create_tenant.py create \
  --tenant-id "new-customer" \
  --name "New Customer Inc" \
  --tier "free"
```

### Upgrade a Tenant

```bash
modal run create_tenant.py upgrade \
  --tenant-id "acme-corp" \
  --tier "enterprise"
```

### View Tenant Info

```bash
modal run create_tenant.py info \
  --tenant-id "acme-corp"
```

### Monitor Logs

```bash
# View live logs
modal app logs multi-tenant-saas
```

## Project Structure

```
multi_tenant_saas/
├── dispatcher.py           # Main API gateway
├── auth_middleware.py      # Authentication
├── rate_limiter.py         # Rate limiting
├── tenant_manager.py       # Tenant management
├── usage_tracker.py        # Usage tracking
├── volume_manager.py       # Storage management
├── worker_functions.py     # Business logic
├── create_tenant.py        # CLI tool
├── config.py              # Configuration
└── tests/                 # Test suite
```

## Customization

### Add Your Business Logic

Edit `worker_functions.py` to add your custom processing:

```python
@app.function()
def my_custom_worker(tenant_id: str, data: dict):
    # Your custom logic here
    vm = create_volume_manager(tenant_id)
    
    # Access tenant data
    vm.write_file("result.json", {"status": "processed"})
    
    return {"success": True}
```

### Adjust Rate Limits

Edit `config.py` to change rate limits:

```python
RATE_LIMIT_CONFIG = {
    "free": {
        "requests_per_hour": 100,  # Change this
        "burst_limit": 20,
    },
    # ...
}
```

### Add GPU Functions

```python
@app.function(gpu="A100")
def gpu_worker(tenant_id: str, data: dict):
    # GPU-accelerated processing
    pass
```

## Troubleshooting

### "JWT_SECRET_KEY not found"
- Create the secret: `modal secret create jwt-secret JWT_SECRET_KEY="your-secret"`

### "Rate limit exceeded"
- Wait for the reset time or upgrade tenant tier
- Check: `curl $API_URL/rate-limit -H "Authorization: Bearer $API_KEY"`

### "Tenant not found"
- Create tenant: `modal run create_tenant.py create ...`

### Logs not showing
- Run: `modal app logs multi-tenant-saas --follow`

## Production Checklist

Before going to production:

- [ ] Generate strong JWT secret (32+ characters)
- [ ] Set up monitoring and alerting
- [ ] Configure proper rate limits for your SLA
- [ ] Set up backup strategy for volumes
- [ ] Implement audit logging
- [ ] Add custom domain to Modal app
- [ ] Set up CI/CD pipeline
- [ ] Create tenant onboarding workflow
- [ ] Test disaster recovery procedures

## Next Steps

1. **Customize Workers** - Add your business logic in `worker_functions.py`
2. **Add Webhooks** - Implement event notifications
3. **Build Frontend** - Create a dashboard for tenants
4. **Add Analytics** - Track and visualize usage patterns
5. **Integrate Billing** - Connect to Stripe or similar

## Support

- 📖 [Full Documentation](README.md)
- 🐛 [Report Issues](https://github.com/modal-labs/modal-examples/issues)
- 💬 [Modal Community](https://modal.com/community)
- 📚 [Modal Docs](https://modal.com/docs)

