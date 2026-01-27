"""
Example Usage - Demonstrates how to use the Multi-Tenant SaaS system.
"""
import json
import time

import requests

# Configuration
API_BASE_URL = "https://your-workspace--multi-tenant-saas-dispatcher.modal.run"

# Replace with your actual API key from tenant creation
API_KEY = "mt_eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."


def example_basic_request():
    """Example: Make a basic request to process data."""
    print("\n" + "="*60)
    print("Example 1: Basic Request")
    print("="*60)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "task_type": "analyze",
        "data": {
            "items": ["apple", "banana", "cherry", "apple", "banana"]
        }
    }

    response = requests.post(
        f"{API_BASE_URL}/process",
        headers=headers,
        json=data,
    )

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    # Check rate limit headers
    print("\nRate Limit Info:")
    print(f"  Limit: {response.headers.get('X-RateLimit-Limit')}")
    print(f"  Remaining: {response.headers.get('X-RateLimit-Remaining')}")
    print(f"  Reset: {response.headers.get('X-RateLimit-Reset')}")


def example_batch_processing():
    """Example: Process multiple tasks in batch."""
    print("\n" + "="*60)
    print("Example 2: Batch Processing")
    print("="*60)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "tasks": [
            {
                "task_type": "analyze",
                "data": {"items": [1, 2, 3, 4, 5]}
            },
            {
                "task_type": "transform",
                "data": {
                    "items": ["hello", "world"],
                    "operation": "uppercase"
                }
            },
            {
                "task_type": "aggregate",
                "data": {}
            }
        ]
    }

    response = requests.post(
        f"{API_BASE_URL}/batch",
        headers=headers,
        json=data,
    )

    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Batch Size: {result.get('batch_size')}")
    print(f"Results: {json.dumps(result.get('results', [])[:2], indent=2)}")  # First 2 results


def example_ml_inference():
    """Example: Run ML model inference."""
    print("\n" + "="*60)
    print("Example 3: ML Inference")
    print("="*60)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model_name": "sentiment-analyzer",
        "inputs": {
            "text": "This is a great product! I love it."
        }
    }

    response = requests.post(
        f"{API_BASE_URL}/inference",
        headers=headers,
        json=data,
    )

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def example_check_usage():
    """Example: Check usage statistics."""
    print("\n" + "="*60)
    print("Example 4: Usage Statistics")
    print("="*60)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }

    response = requests.get(
        f"{API_BASE_URL}/usage",
        headers=headers,
    )

    print(f"Status: {response.status_code}")
    usage = response.json()
    print(f"Tenant ID: {usage.get('tenant_id')}")
    print(f"Date: {usage.get('date')}")
    print(f"Usage: {json.dumps(usage.get('usage', {}), indent=2)}")


def example_rate_limit_status():
    """Example: Check rate limit status."""
    print("\n" + "="*60)
    print("Example 5: Rate Limit Status")
    print("="*60)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }

    response = requests.get(
        f"{API_BASE_URL}/rate-limit",
        headers=headers,
    )

    print(f"Status: {response.status_code}")
    status = response.json()
    print(f"Tenant ID: {status.get('tenant_id')}")
    print(f"Tier: {status.get('tier')}")
    print(f"Rate Limit: {json.dumps(status.get('rate_limit', {}), indent=2)}")


def example_tenant_info():
    """Example: Get tenant information."""
    print("\n" + "="*60)
    print("Example 6: Tenant Information")
    print("="*60)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }

    response = requests.get(
        f"{API_BASE_URL}/tenant",
        headers=headers,
    )

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def example_error_handling():
    """Example: Proper error handling."""
    print("\n" + "="*60)
    print("Example 7: Error Handling")
    print("="*60)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # Try invalid task type
    data = {
        "task_type": "invalid_task",
        "data": {}
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/process",
            headers=headers,
            json=data,
        )

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.json()}")
        else:
            print(f"Success: {response.json()}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


def example_rate_limit_exceeded():
    """Example: Handling rate limit errors."""
    print("\n" + "="*60)
    print("Example 8: Rate Limit Exceeded")
    print("="*60)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "task_type": "analyze",
        "data": {"items": [1, 2, 3]}
    }

    # Make many requests quickly
    for i in range(5):
        response = requests.post(
            f"{API_BASE_URL}/process",
            headers=headers,
            json=data,
        )

        if response.status_code == 429:
            error_data = response.json()
            retry_after = error_data.get('retry_after', 60)
            print(f"Rate limit exceeded! Retry after {retry_after} seconds")
            print(f"Error: {error_data.get('error')}")
            break
        else:
            print(f"Request {i+1}: {response.status_code}")
            time.sleep(0.1)


def example_python_sdk_style():
    """Example: SDK-style wrapper for easier usage."""
    print("\n" + "="*60)
    print("Example 9: SDK-Style Wrapper")
    print("="*60)

    class MultiTenantClient:
        """Simple SDK wrapper for the API."""

        def __init__(self, api_key: str, base_url: str):
            self.api_key = api_key
            self.base_url = base_url
            self.session = requests.Session()
            self.session.headers.update({
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            })

        def process(self, task_type: str, data: dict):
            """Process a single task."""
            response = self.session.post(
                f"{self.base_url}/process",
                json={"task_type": task_type, "data": data},
            )
            response.raise_for_status()
            return response.json()

        def batch(self, tasks: list):
            """Process multiple tasks."""
            response = self.session.post(
                f"{self.base_url}/batch",
                json={"tasks": tasks},
            )
            response.raise_for_status()
            return response.json()

        def inference(self, model_name: str, inputs: dict):
            """Run ML inference."""
            response = self.session.post(
                f"{self.base_url}/inference",
                json={"model_name": model_name, "inputs": inputs},
            )
            response.raise_for_status()
            return response.json()

        def get_usage(self, date: str = None):
            """Get usage statistics."""
            params = {"date": date} if date else {}
            response = self.session.get(
                f"{self.base_url}/usage",
                params=params,
            )
            response.raise_for_status()
            return response.json()

    # Use the client
    client = MultiTenantClient(API_KEY, API_BASE_URL)

    try:
        # Process a task
        result = client.process("analyze", {"items": [1, 2, 3]})
        print(f"Process result: {result.get('success')}")

        # Get usage
        usage = client.get_usage()
        print(f"Usage: {usage.get('usage', {}).get('requests_count')} requests")

    except requests.exceptions.HTTPError as e:
        print(f"API Error: {e}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Multi-Tenant SaaS API Examples")
    print("=" * 60)
    print(f"\nAPI Base URL: {API_BASE_URL}")
    print(f"API Key: {API_KEY[:20]}...")
    print("\nNote: Update API_KEY and API_BASE_URL at the top of this file")

    # Run examples
    try:
        # Basic examples
        example_basic_request()
        time.sleep(0.5)

        example_batch_processing()
        time.sleep(0.5)

        example_ml_inference()
        time.sleep(0.5)

        # Info examples
        example_check_usage()
        time.sleep(0.5)

        example_rate_limit_status()
        time.sleep(0.5)

        example_tenant_info()
        time.sleep(0.5)

        # Error handling
        example_error_handling()
        time.sleep(0.5)

        # SDK style
        example_python_sdk_style()

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nMake sure to:")
        print("1. Deploy the dispatcher: modal deploy dispatcher.py")
        print("2. Create a tenant: modal run create_tenant.py create --tenant-id test --name Test")
        print("3. Update API_KEY and API_BASE_URL in this file")


if __name__ == "__main__":
    main()
