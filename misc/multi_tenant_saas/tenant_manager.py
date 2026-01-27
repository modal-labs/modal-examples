"""
Tenant Manager - Handles tenant provisioning, isolation, and lifecycle.
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import modal
from config import (
    AUTO_CREATE_VOLUMES,
    DEFAULT_TIER,
    TENANT_METADATA_DICT,
    get_dict_key,
    get_resource_limits,
    get_volume_name,
)


class TenantNotFoundError(Exception):
    """Raised when tenant does not exist."""
    pass


class TenantManager:
    """Manages tenant lifecycle, provisioning, and metadata."""

    def __init__(self, metadata_dict: modal.Dict):
        """
        Initialize tenant manager.

        Args:
            metadata_dict: Modal Dict for storing tenant metadata
        """
        self.metadata_dict = metadata_dict

    async def create_tenant(
        self,
        tenant_id: str,
        name: str,
        tier: str = DEFAULT_TIER,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new tenant with isolated resources.

        Args:
            tenant_id: Unique tenant identifier
            name: Tenant display name
            tier: Subscription tier
            metadata: Additional tenant metadata

        Returns:
            Created tenant information
        """
        # Check if tenant already exists
        key = get_dict_key(tenant_id, "metadata")

        try:
            existing = await self.metadata_dict.get.aio(key)
            if existing:
                raise ValueError(f"Tenant {tenant_id} already exists")
        except KeyError:
            pass  # Tenant doesn't exist, which is what we want

        # Create tenant metadata
        tenant_data = {
            "tenant_id": tenant_id,
            "name": name,
            "tier": tier,
            "created_at": datetime.utcnow().isoformat(),
            "status": "active",
            "metadata": metadata or {},
            "resource_limits": get_resource_limits(tier),
        }

        # Store tenant metadata
        await self.metadata_dict.put.aio(key, json.dumps(tenant_data))

        # Create tenant volumes if auto-create is enabled
        if AUTO_CREATE_VOLUMES:
            await self._create_tenant_volumes(tenant_id)

        return tenant_data

    async def _create_tenant_volumes(self, tenant_id: str) -> None:
        """
        Create isolated volumes for a tenant.

        Args:
            tenant_id: Tenant identifier
        """
        # Create data volume for tenant
        data_volume_name = get_volume_name(tenant_id, "data")
        modal.Volume.from_name(
            data_volume_name,
            create_if_missing=True,
        )

        # Create cache volume for tenant (optional)
        cache_volume_name = get_volume_name(tenant_id, "cache")
        modal.Volume.from_name(
            cache_volume_name,
            create_if_missing=True,
        )

        print(f"Created volumes for tenant {tenant_id}:")
        print(f"  - Data: {data_volume_name}")
        print(f"  - Cache: {cache_volume_name}")

    async def get_tenant(self, tenant_id: str) -> Dict[str, Any]:
        """
        Retrieve tenant information.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Tenant metadata

        Raises:
            TenantNotFoundError: If tenant doesn't exist
        """
        key = get_dict_key(tenant_id, "metadata")

        try:
            data = await self.metadata_dict.get.aio(key)
            return json.loads(data)
        except KeyError:
            raise TenantNotFoundError(f"Tenant {tenant_id} not found")

    async def update_tenant(
        self,
        tenant_id: str,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update tenant information.

        Args:
            tenant_id: Tenant identifier
            updates: Fields to update

        Returns:
            Updated tenant metadata
        """
        # Get existing tenant
        tenant = await self.get_tenant(tenant_id)

        # Apply updates
        tenant.update(updates)
        tenant["updated_at"] = datetime.utcnow().isoformat()

        # Save updated tenant
        key = get_dict_key(tenant_id, "metadata")
        await self.metadata_dict.put.aio(key, json.dumps(tenant))

        return tenant

    async def delete_tenant(self, tenant_id: str) -> None:
        """
        Delete a tenant and cleanup resources.

        Args:
            tenant_id: Tenant identifier
        """
        # Mark tenant as deleted
        tenant = await self.get_tenant(tenant_id)
        tenant["status"] = "deleted"
        tenant["deleted_at"] = datetime.utcnow().isoformat()

        key = get_dict_key(tenant_id, "metadata")
        await self.metadata_dict.put.aio(key, json.dumps(tenant))

        # Note: Volume deletion should be done carefully in production
        # You may want to archive data first
        print(f"Tenant {tenant_id} marked as deleted. Manual volume cleanup required.")

    async def list_tenants(
        self,
        status: Optional[str] = "active",
    ) -> List[Dict[str, Any]]:
        """
        List all tenants with optional status filter.

        Args:
            status: Filter by status (active, deleted, suspended)

        Returns:
            List of tenant metadata
        """
        # This is a simplified implementation
        # In production, you might want to maintain an index
        tenants = []

        # Note: Modal Dict doesn't have a native list operation
        # You would need to maintain a separate index or use a database
        # For this example, we'll return an empty list with a note
        print("Note: List operation requires maintaining a tenant index")
        return tenants

    def get_tenant_volumes(self, tenant_id: str) -> Dict[str, modal.Volume]:
        """
        Get Modal Volume objects for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Dictionary mapping volume type to Volume object
        """
        return {
            "data": modal.Volume.from_name(get_volume_name(tenant_id, "data")),
            "cache": modal.Volume.from_name(get_volume_name(tenant_id, "cache")),
        }

    def get_volume_mounts(self, tenant_id: str) -> Dict[str, modal.Volume]:
        """
        Get volume mount configuration for Modal functions.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Dictionary suitable for Modal function volumes parameter
        """
        volumes = self.get_tenant_volumes(tenant_id)

        return {
            "/tenant-data": volumes["data"],
            "/tenant-cache": volumes["cache"],
        }

    async def upgrade_tenant(self, tenant_id: str, new_tier: str) -> Dict[str, Any]:
        """
        Upgrade a tenant to a new tier.

        Args:
            tenant_id: Tenant identifier
            new_tier: New subscription tier

        Returns:
            Updated tenant metadata
        """
        return await self.update_tenant(
            tenant_id,
            {
                "tier": new_tier,
                "resource_limits": get_resource_limits(new_tier),
                "tier_changed_at": datetime.utcnow().isoformat(),
            }
        )

    async def suspend_tenant(self, tenant_id: str, reason: str) -> Dict[str, Any]:
        """
        Suspend a tenant's access.

        Args:
            tenant_id: Tenant identifier
            reason: Reason for suspension

        Returns:
            Updated tenant metadata
        """
        return await self.update_tenant(
            tenant_id,
            {
                "status": "suspended",
                "suspension_reason": reason,
                "suspended_at": datetime.utcnow().isoformat(),
            }
        )

    async def reactivate_tenant(self, tenant_id: str) -> Dict[str, Any]:
        """
        Reactivate a suspended tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Updated tenant metadata
        """
        return await self.update_tenant(
            tenant_id,
            {
                "status": "active",
                "reactivated_at": datetime.utcnow().isoformat(),
            }
        )


def create_tenant_manager() -> TenantManager:
    """
    Create a tenant manager instance with Modal resources.

    Returns:
        Configured TenantManager instance
    """
    metadata_dict = modal.Dict.from_name(
        TENANT_METADATA_DICT,
        create_if_missing=True,
    )

    return TenantManager(metadata_dict)


# Example usage
"""
# In a Modal function:
@app.function()
async def setup_new_tenant(tenant_id: str, name: str):
    manager = create_tenant_manager()

    # Create tenant with isolated resources
    tenant = await manager.create_tenant(
        tenant_id=tenant_id,
        name=name,
        tier="pro",
        metadata={"company": "Acme Corp", "industry": "Tech"}
    )

    return tenant

# Later, get volume mounts for tenant-specific operations:
@app.function(volumes=manager.get_volume_mounts(tenant_id))
def process_tenant_data(tenant_id: str, data: dict):
    # This function has access to tenant-specific volumes only
    with open("/tenant-data/input.json", "w") as f:
        json.dump(data, f)

    # Process data...
    result = process(data)

    # Save to tenant volume
    with open("/tenant-data/output.json", "w") as f:
        json.dump(result, f)

    volumes["data"].commit()
    return result
"""
