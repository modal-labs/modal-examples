"""
Tests for tenant isolation.
"""
import shutil
import tempfile
from pathlib import Path

import pytest
from config import get_volume_name
from volume_manager import VolumeManager


@pytest.fixture
def temp_volume_path():
    """Create a temporary directory to simulate volume mount."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


def test_volume_name_generation():
    """Test that volume names are properly namespaced per tenant."""
    tenant1_vol = get_volume_name("tenant-1", "data")
    tenant2_vol = get_volume_name("tenant-2", "data")

    assert tenant1_vol != tenant2_vol
    assert "tenant-1" in tenant1_vol
    assert "tenant-2" in tenant2_vol
    assert tenant1_vol.startswith("tenant-")
    assert tenant2_vol.startswith("tenant-")


def test_volume_manager_initialization():
    """Test VolumeManager initialization."""
    vm1 = VolumeManager("tenant-1")
    vm2 = VolumeManager("tenant-2")

    assert vm1.tenant_id == "tenant-1"
    assert vm2.tenant_id == "tenant-2"
    assert vm1.tenant_id != vm2.tenant_id


def test_tenant_path_isolation(temp_volume_path, monkeypatch):
    """Test that tenant paths are isolated."""
    # Mock the volume mount path
    monkeypatch.setattr("volume_manager.VOLUME_MOUNT_PATH", str(temp_volume_path))

    vm1 = VolumeManager("tenant-1")
    vm2 = VolumeManager("tenant-2")

    # Create tenant directories
    path1 = vm1.ensure_tenant_directory("data")
    path2 = vm2.ensure_tenant_directory("data")

    # Paths should be different
    assert path1 != path2
    assert "tenant-1" in str(path1) or path1.parent.name == "data"
    assert "tenant-2" in str(path2) or path2.parent.name == "data"


def test_file_operations_isolation(temp_volume_path, monkeypatch):
    """Test that file operations are isolated between tenants."""
    monkeypatch.setattr("volume_manager.VOLUME_MOUNT_PATH", str(temp_volume_path))

    vm1 = VolumeManager("tenant-1")
    vm2 = VolumeManager("tenant-2")

    # Create test directories
    (temp_volume_path / "data").mkdir(exist_ok=True)

    # Tenant 1 writes a file
    vm1.write_file("secret.txt", "tenant-1-data", volume_type="data", commit=False)

    # Tenant 2 writes a file with same name
    vm2.write_file("secret.txt", "tenant-2-data", volume_type="data", commit=False)

    # Each tenant should read their own file
    data1 = vm1.read_file("secret.txt", volume_type="data")
    data2 = vm2.read_file("secret.txt", volume_type="data")

    assert data1 == "tenant-1-data"
    assert data2 == "tenant-2-data"
    assert data1 != data2


def test_list_files_isolation(temp_volume_path, monkeypatch):
    """Test that file listing is isolated."""
    monkeypatch.setattr("volume_manager.VOLUME_MOUNT_PATH", str(temp_volume_path))

    vm1 = VolumeManager("tenant-1")
    vm2 = VolumeManager("tenant-2")

    # Create test directories
    (temp_volume_path / "data").mkdir(exist_ok=True)

    # Tenant 1 creates multiple files
    vm1.write_file("file1.txt", "data1", volume_type="data", commit=False)
    vm1.write_file("file2.txt", "data2", volume_type="data", commit=False)

    # Tenant 2 creates different files
    vm2.write_file("file3.txt", "data3", volume_type="data", commit=False)

    # Each tenant should only see their own files
    files1 = vm1.list_files(volume_type="data")
    files2 = vm2.list_files(volume_type="data")

    assert "file1.txt" in files1
    assert "file2.txt" in files1
    assert "file3.txt" not in files1

    assert "file3.txt" in files2
    assert "file1.txt" not in files2
    assert "file2.txt" not in files2


def test_delete_file_isolation(temp_volume_path, monkeypatch):
    """Test that file deletion is isolated."""
    monkeypatch.setattr("volume_manager.VOLUME_MOUNT_PATH", str(temp_volume_path))

    vm1 = VolumeManager("tenant-1")
    vm2 = VolumeManager("tenant-2")

    # Create test directories
    (temp_volume_path / "data").mkdir(exist_ok=True)

    # Both tenants create files with same name
    vm1.write_file("data.txt", "tenant-1", volume_type="data", commit=False)
    vm2.write_file("data.txt", "tenant-2", volume_type="data", commit=False)

    # Tenant 1 deletes their file
    deleted = vm1.delete_file("data.txt", volume_type="data", commit=False)
    assert deleted is True

    # Tenant 1's file should be gone
    assert "data.txt" not in vm1.list_files(volume_type="data")

    # Tenant 2's file should still exist
    assert "data.txt" in vm2.list_files(volume_type="data")
    data2 = vm2.read_file("data.txt", volume_type="data")
    assert data2 == "tenant-2"


def test_json_file_operations(temp_volume_path, monkeypatch):
    """Test JSON file operations with isolation."""
    monkeypatch.setattr("volume_manager.VOLUME_MOUNT_PATH", str(temp_volume_path))

    vm1 = VolumeManager("tenant-1")
    vm2 = VolumeManager("tenant-2")

    # Create test directories
    (temp_volume_path / "data").mkdir(exist_ok=True)

    # Write JSON data
    data1 = {"tenant": "tenant-1", "value": 100}
    data2 = {"tenant": "tenant-2", "value": 200}

    vm1.write_file("config.json", data1, volume_type="data", commit=False)
    vm2.write_file("config.json", data2, volume_type="data", commit=False)

    # Read and verify
    read1 = vm1.read_file("config.json", volume_type="data", as_json=True)
    read2 = vm2.read_file("config.json", volume_type="data", as_json=True)

    assert read1["tenant"] == "tenant-1"
    assert read1["value"] == 100
    assert read2["tenant"] == "tenant-2"
    assert read2["value"] == 200


def test_volume_stats_isolation(temp_volume_path, monkeypatch):
    """Test that volume statistics are isolated."""
    monkeypatch.setattr("volume_manager.VOLUME_MOUNT_PATH", str(temp_volume_path))

    vm1 = VolumeManager("tenant-1")
    vm2 = VolumeManager("tenant-2")

    # Create test directories
    (temp_volume_path / "data").mkdir(exist_ok=True)

    # Tenant 1 creates 3 files
    vm1.write_file("file1.txt", "a" * 100, volume_type="data", commit=False)
    vm1.write_file("file2.txt", "b" * 200, volume_type="data", commit=False)
    vm1.write_file("file3.txt", "c" * 300, volume_type="data", commit=False)

    # Tenant 2 creates 2 files
    vm2.write_file("file1.txt", "x" * 150, volume_type="data", commit=False)
    vm2.write_file("file2.txt", "y" * 250, volume_type="data", commit=False)

    # Get stats
    stats1 = vm1.get_volume_stats(volume_type="data")
    stats2 = vm2.get_volume_stats(volume_type="data")

    # Each tenant should only see their own stats
    assert stats1["file_count"] == 3
    assert stats2["file_count"] == 2
    assert stats1["total_size_bytes"] == 600
    assert stats2["total_size_bytes"] == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
