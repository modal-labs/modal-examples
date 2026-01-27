"""
Volume Manager - Handles tenant-scoped volume operations and namespacing.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal
from config import (
    VOLUME_MOUNT_PATH,
    get_volume_name,
)


class VolumeManager:
    """
    Manages tenant-scoped volume operations with automatic namespacing.
    Ensures complete isolation between tenant data.
    """

    def __init__(self, tenant_id: str):
        """
        Initialize volume manager for a specific tenant.

        Args:
            tenant_id: Tenant identifier
        """
        self.tenant_id = tenant_id
        self._volumes = {}

    def get_volume(self, volume_type: str = "data") -> modal.Volume:
        """
        Get a Modal Volume for the tenant.

        Args:
            volume_type: Type of volume (data, cache, models, etc.)

        Returns:
            Modal Volume object for the tenant
        """
        if volume_type not in self._volumes:
            volume_name = get_volume_name(self.tenant_id, volume_type)
            self._volumes[volume_type] = modal.Volume.from_name(
                volume_name,
                create_if_missing=True,
            )

        return self._volumes[volume_type]

    def get_mount_config(
        self,
        volume_types: Optional[List[str]] = None,
    ) -> Dict[str, modal.Volume]:
        """
        Get volume mount configuration for Modal functions.

        Args:
            volume_types: List of volume types to mount (defaults to ["data"])

        Returns:
            Dictionary mapping mount paths to Volume objects
        """
        if volume_types is None:
            volume_types = ["data"]

        mounts = {}
        for vtype in volume_types:
            volume = self.get_volume(vtype)
            mount_path = f"{VOLUME_MOUNT_PATH}/{vtype}"
            mounts[mount_path] = volume

        return mounts

    def get_tenant_path(self, volume_type: str = "data") -> Path:
        """
        Get the base path for tenant data in a mounted volume.

        Args:
            volume_type: Type of volume

        Returns:
            Path object for tenant's directory
        """
        return Path(VOLUME_MOUNT_PATH) / volume_type

    def ensure_tenant_directory(self, volume_type: str = "data") -> Path:
        """
        Ensure tenant directory exists in volume.

        Args:
            volume_type: Type of volume

        Returns:
            Path to tenant directory
        """
        tenant_path = self.get_tenant_path(volume_type)
        tenant_path.mkdir(parents=True, exist_ok=True)
        return tenant_path

    def write_file(
        self,
        filename: str,
        content: Any,
        volume_type: str = "data",
        commit: bool = True,
    ) -> Path:
        """
        Write a file to tenant's volume.

        Args:
            filename: Name of file to write
            content: Content to write (str, bytes, or dict for JSON)
            volume_type: Type of volume
            commit: Whether to commit changes immediately

        Returns:
            Path to written file
        """
        tenant_path = self.ensure_tenant_directory(volume_type)
        file_path = tenant_path / filename

        # Handle different content types
        if isinstance(content, dict):
            with open(file_path, 'w') as f:
                json.dump(content, f, indent=2)
        elif isinstance(content, str):
            with open(file_path, 'w') as f:
                f.write(content)
        elif isinstance(content, bytes):
            with open(file_path, 'wb') as f:
                f.write(content)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

        if commit:
            self.commit_volume(volume_type)

        return file_path

    def read_file(
        self,
        filename: str,
        volume_type: str = "data",
        as_json: bool = False,
    ) -> Any:
        """
        Read a file from tenant's volume.

        Args:
            filename: Name of file to read
            volume_type: Type of volume
            as_json: Whether to parse as JSON

        Returns:
            File content
        """
        tenant_path = self.get_tenant_path(volume_type)
        file_path = tenant_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        if as_json:
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            with open(file_path, 'r') as f:
                return f.read()

    def list_files(
        self,
        volume_type: str = "data",
        pattern: str = "*",
    ) -> List[str]:
        """
        List files in tenant's volume.

        Args:
            volume_type: Type of volume
            pattern: Glob pattern for filtering

        Returns:
            List of filenames
        """
        tenant_path = self.get_tenant_path(volume_type)

        if not tenant_path.exists():
            return []

        files = [f.name for f in tenant_path.glob(pattern) if f.is_file()]
        return sorted(files)

    def delete_file(
        self,
        filename: str,
        volume_type: str = "data",
        commit: bool = True,
    ) -> bool:
        """
        Delete a file from tenant's volume.

        Args:
            filename: Name of file to delete
            volume_type: Type of volume
            commit: Whether to commit changes immediately

        Returns:
            True if file was deleted
        """
        tenant_path = self.get_tenant_path(volume_type)
        file_path = tenant_path / filename

        if file_path.exists():
            file_path.unlink()
            if commit:
                self.commit_volume(volume_type)
            return True

        return False

    def commit_volume(self, volume_type: str = "data") -> None:
        """
        Commit changes to a volume.

        Args:
            volume_type: Type of volume to commit
        """
        volume = self.get_volume(volume_type)
        volume.commit()

    def reload_volume(self, volume_type: str = "data") -> None:
        """
        Reload a volume to get latest changes.

        Args:
            volume_type: Type of volume to reload
        """
        volume = self.get_volume(volume_type)
        volume.reload()

    def get_volume_stats(self, volume_type: str = "data") -> Dict[str, Any]:
        """
        Get statistics about tenant's volume usage.

        Args:
            volume_type: Type of volume

        Returns:
            Dictionary with volume statistics
        """
        tenant_path = self.get_tenant_path(volume_type)

        if not tenant_path.exists():
            return {
                "exists": False,
                "file_count": 0,
                "total_size_bytes": 0,
            }

        file_count = 0
        total_size = 0

        for file_path in tenant_path.rglob("*"):
            if file_path.is_file():
                file_count += 1
                total_size += file_path.stat().st_size

        return {
            "exists": True,
            "file_count": file_count,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "tenant_id": self.tenant_id,
            "volume_type": volume_type,
        }

    def cleanup_old_files(
        self,
        volume_type: str = "data",
        days_old: int = 30,
        commit: bool = True,
    ) -> int:
        """
        Delete files older than specified days.

        Args:
            volume_type: Type of volume
            days_old: Delete files older than this many days
            commit: Whether to commit changes

        Returns:
            Number of files deleted
        """
        import time

        tenant_path = self.get_tenant_path(volume_type)
        if not tenant_path.exists():
            return 0

        cutoff_time = time.time() - (days_old * 24 * 3600)
        deleted_count = 0

        for file_path in tenant_path.rglob("*"):
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1

        if commit and deleted_count > 0:
            self.commit_volume(volume_type)

        return deleted_count

    def copy_to_volume(
        self,
        local_path: str,
        volume_filename: str,
        volume_type: str = "data",
        commit: bool = True,
    ) -> Path:
        """
        Copy a local file to tenant's volume.

        Args:
            local_path: Path to local file
            volume_filename: Destination filename in volume
            volume_type: Type of volume
            commit: Whether to commit changes

        Returns:
            Path to file in volume
        """
        with open(local_path, 'rb') as f:
            content = f.read()

        return self.write_file(volume_filename, content, volume_type, commit)

    def copy_from_volume(
        self,
        volume_filename: str,
        local_path: str,
        volume_type: str = "data",
    ) -> None:
        """
        Copy a file from tenant's volume to local path.

        Args:
            volume_filename: Filename in volume
            local_path: Destination local path
            volume_type: Type of volume
        """
        tenant_path = self.get_tenant_path(volume_type)
        file_path = tenant_path / volume_filename

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {volume_filename}")

        with open(file_path, 'rb') as src:
            with open(local_path, 'wb') as dst:
                dst.write(src.read())


def create_volume_manager(tenant_id: str) -> VolumeManager:
    """
    Create a volume manager for a tenant.

    Args:
        tenant_id: Tenant identifier

    Returns:
        Configured VolumeManager instance
    """
    return VolumeManager(tenant_id)


# Example usage in a Modal function
"""
@app.function(
    volumes=create_volume_manager("customer-123").get_mount_config(["data", "cache"])
)
def process_with_volumes(tenant_id: str, data: dict):
    # Create volume manager
    vm = create_volume_manager(tenant_id)

    # Write input data
    vm.write_file("input.json", data, volume_type="data")

    # Process data
    result = process(data)

    # Save results
    vm.write_file("output.json", result, volume_type="data", commit=True)

    # List all files
    files = vm.list_files(volume_type="data")
    print(f"Files in volume: {files}")

    # Get stats
    stats = vm.get_volume_stats(volume_type="data")
    print(f"Volume usage: {stats['total_size_mb']:.2f} MB")

    return result
"""
