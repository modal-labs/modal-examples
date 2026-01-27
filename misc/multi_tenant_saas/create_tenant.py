"""
Create Tenant CLI - Tool to create new tenants and generate API keys.
"""
import argparse
import sys
from typing import Optional

import modal
from auth_middleware import create_auth_middleware
from config import MODAL_APP_NAME, MODAL_IMAGE_PACKAGES
from tenant_manager import create_tenant_manager

# Create Modal image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*MODAL_IMAGE_PACKAGES)
)

# Create Modal app
app = modal.App(f"{MODAL_APP_NAME}-admin", image=image)


@app.function(secrets=[modal.Secret.from_name("jwt-secret")])
async def create_tenant_remote(
    tenant_id: str,
    name: str,
    tier: str = "free",
    email: Optional[str] = None,
    company: Optional[str] = None,
) -> dict:
    """
    Create a new tenant with isolated resources.

    Args:
        tenant_id: Unique tenant identifier
        name: Tenant display name
        tier: Subscription tier (free, pro, enterprise)
        email: Contact email
        company: Company name

    Returns:
        Tenant info and API key
    """
    # Create tenant manager
    tenant_manager = create_tenant_manager()

    # Prepare metadata
    metadata = {
        "name": name,
        "is_admin": False,
    }
    if email:
        metadata["email"] = email
    if company:
        metadata["company"] = company

    # Create tenant
    try:
        tenant = await tenant_manager.create_tenant(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            metadata=metadata,
        )

        print(f"✓ Created tenant: {tenant_id}")
        print(f"  Name: {name}")
        print(f"  Tier: {tier}")
        print(f"  Status: {tenant['status']}")

    except ValueError as e:
        print(f"✗ Error: {e}")
        return {"error": str(e)}

    # Generate API key
    auth = create_auth_middleware()
    api_key = auth.generate_token(
        tenant_id=tenant_id,
        tier=tier,
        metadata=metadata,
        expiry_hours=8760,  # 1 year
    )

    print("\n✓ Generated API key (expires in 1 year)")

    return {
        "tenant_id": tenant_id,
        "name": name,
        "tier": tier,
        "api_key": api_key,
        "tenant": tenant,
    }


@app.function(secrets=[modal.Secret.from_name("jwt-secret")])
async def list_tenants_remote():
    """
    List all tenants (admin function).

    Returns:
        List of tenants
    """
    tenant_manager = create_tenant_manager()
    tenants = await tenant_manager.list_tenants()

    return tenants


@app.function(secrets=[modal.Secret.from_name("jwt-secret")])
async def delete_tenant_remote(tenant_id: str):
    """
    Delete a tenant (admin function).

    Args:
        tenant_id: Tenant to delete

    Returns:
        Deletion status
    """
    tenant_manager = create_tenant_manager()

    try:
        await tenant_manager.delete_tenant(tenant_id)
        print(f"✓ Tenant {tenant_id} marked as deleted")
        print("  Note: Manual volume cleanup may be required")
        return {"success": True, "tenant_id": tenant_id}
    except Exception as e:
        print(f"✗ Error deleting tenant: {e}")
        return {"error": str(e)}


@app.function(secrets=[modal.Secret.from_name("jwt-secret")])
async def upgrade_tenant_remote(tenant_id: str, new_tier: str):
    """
    Upgrade a tenant to a new tier.

    Args:
        tenant_id: Tenant to upgrade
        new_tier: New subscription tier

    Returns:
        Updated tenant info
    """
    tenant_manager = create_tenant_manager()

    try:
        tenant = await tenant_manager.upgrade_tenant(tenant_id, new_tier)
        print(f"✓ Upgraded tenant {tenant_id} to {new_tier} tier")
        return {"success": True, "tenant": tenant}
    except Exception as e:
        print(f"✗ Error upgrading tenant: {e}")
        return {"error": str(e)}


@app.function(secrets=[modal.Secret.from_name("jwt-secret")])
async def get_tenant_info_remote(tenant_id: str):
    """
    Get information about a tenant.

    Args:
        tenant_id: Tenant identifier

    Returns:
        Tenant information
    """
    tenant_manager = create_tenant_manager()

    try:
        tenant = await tenant_manager.get_tenant(tenant_id)
        return {"success": True, "tenant": tenant}
    except Exception as e:
        return {"error": str(e)}


@app.local_entrypoint()
def main():
    """
    CLI entrypoint for tenant management.
    """
    parser = argparse.ArgumentParser(
        description="Manage tenants for Multi-Tenant SaaS"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Create tenant command
    create_parser = subparsers.add_parser("create", help="Create a new tenant")
    create_parser.add_argument("--tenant-id", required=True, help="Unique tenant ID")
    create_parser.add_argument("--name", required=True, help="Tenant display name")
    create_parser.add_argument(
        "--tier",
        default="free",
        choices=["free", "pro", "enterprise"],
        help="Subscription tier"
    )
    create_parser.add_argument("--email", help="Contact email")
    create_parser.add_argument("--company", help="Company name")

    # List tenants command
    subparsers.add_parser("list", help="List all tenants")

    # Delete tenant command
    delete_parser = subparsers.add_parser("delete", help="Delete a tenant")
    delete_parser.add_argument("--tenant-id", required=True, help="Tenant ID to delete")

    # Upgrade tenant command
    upgrade_parser = subparsers.add_parser("upgrade", help="Upgrade tenant tier")
    upgrade_parser.add_argument("--tenant-id", required=True, help="Tenant ID")
    upgrade_parser.add_argument(
        "--tier",
        required=True,
        choices=["free", "pro", "enterprise"],
        help="New tier"
    )

    # Get tenant info command
    info_parser = subparsers.add_parser("info", help="Get tenant information")
    info_parser.add_argument("--tenant-id", required=True, help="Tenant ID")

    args = parser.parse_args()

    if args.command == "create":
        result = create_tenant_remote.remote(
            tenant_id=args.tenant_id,
            name=args.name,
            tier=args.tier,
            email=args.email,
            company=args.company,
        )

        if "error" not in result:
            print("\n" + "=" * 60)
            print("API KEY (save this, it won't be shown again):")
            print("=" * 60)
            print(result["api_key"])
            print("=" * 60)
            print("\nTest with:")
            print(f'curl -H "Authorization: Bearer {result["api_key"]}" \\')
            print('     https://your-workspace--multi-tenant-saas-dispatcher.modal.run/health')

    elif args.command == "list":
        result = list_tenants_remote.remote()
        print("\nTenants:")
        for tenant in result:
            print(f"  - {tenant.get('tenant_id')}: {tenant.get('name')} ({tenant.get('tier')})")

    elif args.command == "delete":
        result = delete_tenant_remote.remote(args.tenant_id)

    elif args.command == "upgrade":
        result = upgrade_tenant_remote.remote(args.tenant_id, args.tier)

    elif args.command == "info":
        result = get_tenant_info_remote.remote(args.tenant_id)

        if "error" not in result:
            import json
            print("\nTenant Information:")
            print(json.dumps(result["tenant"], indent=2))
        else:
            print(f"✗ Error: {result['error']}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
