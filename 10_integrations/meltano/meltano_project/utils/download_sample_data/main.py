# fetches the tutorial data used by dbt in their tutorials: https://docs.getdbt.com/docs/get-started/getting-started-dbt-core
from pathlib import Path

import requests

output_dir = Path("downloads")
dbt_sample_files = [
    "https://dbt-tutorial-public.s3-us-west-2.amazonaws.com/jaffle_shop_customers.csv",
    "https://dbt-tutorial-public.s3-us-west-2.amazonaws.com/jaffle_shop_orders.csv",
    "https://dbt-tutorial-public.s3-us-west-2.amazonaws.com/stripe_payments.csv",
]


def run():
    output_dir.mkdir(parents=True, exist_ok=True)
    for url in dbt_sample_files:
        content = requests.get(url).content.decode("utf-8-sig")  # stupid bom
        filename = url.rsplit("/", 1)[1]
        print(f"Fetched csv data: {filename}")
        (output_dir / filename).write_bytes(content.encode("utf8"))
