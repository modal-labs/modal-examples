"""
Usage Tracker - Monitor tenant resource consumption for billing and quotas.
"""
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import modal
from config import (
    USAGE_STATS_DICT,
    get_dict_key,
    get_resource_limits,
)


@dataclass
class UsageMetrics:
    """Container for usage metrics."""
    tenant_id: str
    date: str
    requests_count: int = 0
    compute_seconds: float = 0.0
    storage_bytes: int = 0
    data_transfer_bytes: int = 0
    gpu_seconds: float = 0.0
    errors_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UsageMetrics':
        return cls(**data)


class ResourceQuotaExceeded(Exception):
    """Raised when tenant exceeds resource quota."""
    pass


class UsageTracker:
    """
    Track and monitor tenant resource usage for billing and quota enforcement.
    """

    def __init__(self, usage_dict: modal.Dict):
        """
        Initialize usage tracker.

        Args:
            usage_dict: Modal Dict for storing usage metrics
        """
        self.usage_dict = usage_dict

    async def record_request(
        self,
        tenant_id: str,
        compute_seconds: float = 0.0,
        data_transfer_bytes: int = 0,
        gpu_seconds: float = 0.0,
        error: bool = False,
    ) -> UsageMetrics:
        """
        Record a request and its resource usage.

        Args:
            tenant_id: Tenant identifier
            compute_seconds: CPU time consumed
            data_transfer_bytes: Network transfer in bytes
            gpu_seconds: GPU time consumed
            error: Whether the request resulted in an error

        Returns:
            Updated usage metrics
        """
        today = datetime.utcnow().date().isoformat()
        key = get_dict_key(tenant_id, f"usage:{today}")

        # Get or create today's metrics
        try:
            data = await self.usage_dict.get.aio(key)
            metrics = UsageMetrics.from_dict(json.loads(data))
        except KeyError:
            metrics = UsageMetrics(tenant_id=tenant_id, date=today)

        # Update metrics
        metrics.requests_count += 1
        metrics.compute_seconds += compute_seconds
        metrics.data_transfer_bytes += data_transfer_bytes
        metrics.gpu_seconds += gpu_seconds
        if error:
            metrics.errors_count += 1

        # Save updated metrics
        await self.usage_dict.put.aio(key, json.dumps(metrics.to_dict()))

        return metrics

    async def get_usage(
        self,
        tenant_id: str,
        date: Optional[str] = None,
    ) -> UsageMetrics:
        """
        Get usage metrics for a specific date.

        Args:
            tenant_id: Tenant identifier
            date: Date in ISO format (defaults to today)

        Returns:
            Usage metrics for the date
        """
        if date is None:
            date = datetime.utcnow().date().isoformat()

        key = get_dict_key(tenant_id, f"usage:{date}")

        try:
            data = await self.usage_dict.get.aio(key)
            return UsageMetrics.from_dict(json.loads(data))
        except KeyError:
            # No usage data for this date
            return UsageMetrics(tenant_id=tenant_id, date=date)

    async def get_usage_range(
        self,
        tenant_id: str,
        start_date: str,
        end_date: str,
    ) -> List[UsageMetrics]:
        """
        Get usage metrics for a date range.

        Args:
            tenant_id: Tenant identifier
            start_date: Start date in ISO format
            end_date: End date in ISO format

        Returns:
            List of usage metrics for each day in range
        """
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)

        metrics_list = []
        current = start

        while current <= end:
            date_str = current.date().isoformat()
            metrics = await self.get_usage(tenant_id, date_str)
            metrics_list.append(metrics)
            current += timedelta(days=1)

        return metrics_list

    async def get_monthly_usage(
        self,
        tenant_id: str,
        year: int,
        month: int,
    ) -> Dict[str, Any]:
        """
        Get aggregated usage for a month.

        Args:
            tenant_id: Tenant identifier
            year: Year
            month: Month (1-12)

        Returns:
            Aggregated monthly usage
        """
        # Get date range for month
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)

        # Get all daily metrics for the month
        daily_metrics = await self.get_usage_range(
            tenant_id,
            start_date.date().isoformat(),
            end_date.date().isoformat(),
        )

        # Aggregate metrics
        total_requests = sum(m.requests_count for m in daily_metrics)
        total_compute = sum(m.compute_seconds for m in daily_metrics)
        total_transfer = sum(m.data_transfer_bytes for m in daily_metrics)
        total_gpu = sum(m.gpu_seconds for m in daily_metrics)
        total_errors = sum(m.errors_count for m in daily_metrics)

        return {
            "tenant_id": tenant_id,
            "year": year,
            "month": month,
            "total_requests": total_requests,
            "total_compute_seconds": total_compute,
            "total_compute_hours": total_compute / 3600,
            "total_data_transfer_gb": total_transfer / (1024**3),
            "total_gpu_seconds": total_gpu,
            "total_gpu_hours": total_gpu / 3600,
            "total_errors": total_errors,
            "error_rate": total_errors / total_requests if total_requests > 0 else 0,
            "days": len(daily_metrics),
        }

    async def check_quota(
        self,
        tenant_id: str,
        tier: str,
        resource_type: str,
        amount: float,
    ) -> bool:
        """
        Check if tenant has quota available for a resource.

        Args:
            tenant_id: Tenant identifier
            tier: Tenant subscription tier
            resource_type: Type of resource (compute, storage, gpu)
            amount: Amount of resource needed

        Returns:
            True if quota available

        Raises:
            ResourceQuotaExceeded: If quota exceeded
        """
        limits = get_resource_limits(tier)
        today_usage = await self.get_usage(tenant_id)

        if resource_type == "compute":
            limit = limits["max_compute_seconds_per_day"]
            used = today_usage.compute_seconds

            if used + amount > limit:
                raise ResourceQuotaExceeded(
                    f"Daily compute quota exceeded. "
                    f"Used: {used:.1f}s, Limit: {limit}s"
                )

        elif resource_type == "storage":
            limit = limits["max_storage_gb"] * (1024**3)  # Convert to bytes

            if amount > limit:
                raise ResourceQuotaExceeded(
                    f"Storage quota exceeded. "
                    f"Requested: {amount/(1024**3):.2f}GB, Limit: {limits['max_storage_gb']}GB"
                )

        return True

    async def get_cost_estimate(
        self,
        tenant_id: str,
        start_date: str,
        end_date: str,
        pricing: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Estimate costs based on usage.

        Args:
            tenant_id: Tenant identifier
            start_date: Start date
            end_date: End date
            pricing: Pricing per unit (optional)

        Returns:
            Cost breakdown
        """
        # Default pricing (example rates)
        if pricing is None:
            pricing = {
                "compute_per_hour": 0.10,  # $0.10 per compute hour
                "gpu_per_hour": 1.00,      # $1.00 per GPU hour
                "transfer_per_gb": 0.05,   # $0.05 per GB transfer
                "storage_per_gb_month": 0.01,  # $0.01 per GB per month
            }

        # Get usage for date range
        metrics_list = await self.get_usage_range(tenant_id, start_date, end_date)

        # Calculate totals
        total_compute_hours = sum(m.compute_seconds for m in metrics_list) / 3600
        total_gpu_hours = sum(m.gpu_seconds for m in metrics_list) / 3600
        total_transfer_gb = sum(m.data_transfer_bytes for m in metrics_list) / (1024**3)

        # Calculate costs
        compute_cost = total_compute_hours * pricing["compute_per_hour"]
        gpu_cost = total_gpu_hours * pricing["gpu_per_hour"]
        transfer_cost = total_transfer_gb * pricing["transfer_per_gb"]

        total_cost = compute_cost + gpu_cost + transfer_cost

        return {
            "tenant_id": tenant_id,
            "start_date": start_date,
            "end_date": end_date,
            "compute_hours": total_compute_hours,
            "compute_cost": compute_cost,
            "gpu_hours": total_gpu_hours,
            "gpu_cost": gpu_cost,
            "transfer_gb": total_transfer_gb,
            "transfer_cost": transfer_cost,
            "total_cost": total_cost,
            "currency": "USD",
        }

    async def export_usage_report(
        self,
        tenant_id: str,
        month: str,
    ) -> Dict[str, Any]:
        """
        Export a comprehensive usage report for billing.

        Args:
            tenant_id: Tenant identifier
            month: Month in YYYY-MM format

        Returns:
            Comprehensive usage report
        """
        year, month_num = map(int, month.split("-"))

        monthly_usage = await self.get_monthly_usage(tenant_id, year, month_num)

        # Get first and last day of month
        start_date = datetime(year, month_num, 1).date().isoformat()
        if month_num == 12:
            end_date = datetime(year + 1, 1, 1).date() - timedelta(days=1)
        else:
            end_date = datetime(year, month_num + 1, 1).date() - timedelta(days=1)

        cost_estimate = await self.get_cost_estimate(
            tenant_id,
            start_date,
            end_date.isoformat(),
        )

        return {
            "tenant_id": tenant_id,
            "billing_period": month,
            "usage": monthly_usage,
            "costs": cost_estimate,
            "generated_at": datetime.utcnow().isoformat(),
        }


def create_usage_tracker() -> UsageTracker:
    """
    Create a usage tracker instance with Modal resources.

    Returns:
        Configured UsageTracker instance
    """
    usage_dict = modal.Dict.from_name(
        USAGE_STATS_DICT,
        create_if_missing=True,
    )

    return UsageTracker(usage_dict)


# Decorator to automatically track usage
def track_usage(resource_type: str = "compute"):
    """
    Decorator to automatically track resource usage.

    Args:
        resource_type: Type of resource being tracked

    Usage:
        @app.function()
        @track_usage("compute")
        async def my_function(tenant_id: str, tier: str):
            # Usage is automatically tracked
            pass
    """
    def decorator(func):
        async def wrapper(tenant_id: str, tier: str, *args, **kwargs):
            tracker = create_usage_tracker()
            start_time = time.time()
            error = False

            try:
                result = await func(tenant_id, tier, *args, **kwargs)
                return result
            except Exception:
                error = True
                raise
            finally:
                # Record usage
                elapsed = time.time() - start_time
                await tracker.record_request(
                    tenant_id=tenant_id,
                    compute_seconds=elapsed,
                    error=error,
                )

        return wrapper
    return decorator


# Example usage
"""
@app.function()
async def billing_example(tenant_id: str):
    tracker = create_usage_tracker()

    # Record a request
    await tracker.record_request(
        tenant_id=tenant_id,
        compute_seconds=120.5,
        data_transfer_bytes=1024 * 1024 * 100,  # 100 MB
        gpu_seconds=60.0,
    )

    # Get today's usage
    usage = await tracker.get_usage(tenant_id)
    print(f"Requests today: {usage.requests_count}")
    print(f"Compute time: {usage.compute_seconds}s")

    # Get monthly report
    report = await tracker.export_usage_report(tenant_id, "2026-01")
    print(f"Monthly cost: ${report['costs']['total_cost']:.2f}")
"""
