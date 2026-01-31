# ---
# cmd: ["python", "misc/kafka_microbatch_etl.py", "--batch=25", "--timeout-s=5", "--local=true"]
# runtimes: ["runc", "gvisor"]
# ---
#
# # Kafka micro-batch ETL (bounded)
#
# Polls up to N Kafka messages (or time limit), applies a tiny transform,
# POSTs the batch to a REST sink, then exits.
#
# Intended for ETL/backfills/periodic jobs — NOT continuous stream processing
# (e.g. Flink / Kafka Streams). This example is intentionally single-worker.

import json
import os
import time

import modal
import requests
from confluent_kafka import Consumer

image = modal.Image.debian_slim().pip_install("confluent-kafka", "requests")
app = modal.App(
    "kafka-microbatch-etl",
    image=image,
    secrets=[modal.Secret.from_name("kafka-etl-remote-v2")],
)


@app.function()
def etl(batch: int = 100, timeout_s: int = 5):
    c = Consumer(
        {
            "bootstrap.servers": os.environ["KAFKA_BOOTSTRAP"],
            "group.id": os.getenv("KAFKA_GROUP", "modal-microbatch"),
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,  # keep example safe/simple
            # Confluent Cloud auth:
            "security.protocol": "SASL_SSL",
            "sasl.mechanism": "PLAIN",
            "sasl.username": os.environ["KAFKA_API_KEY"],
            "sasl.password": os.environ["KAFKA_API_SECRET"],
        }
    )
    c.subscribe([os.environ["KAFKA_TOPIC"]])

    rows, end = [], time.time() + timeout_s
    while len(rows) < batch and time.time() < end:
        m = c.poll(0.5)
        if not m or m.error():
            continue
        rows.append(
            {
                "ts": m.timestamp()[1],
                "payload": json.loads(m.value().decode("utf-8")),
            }
        )
    c.close()

    if rows:
        url = os.getenv("SINK_URL", "https://httpbin.org/post")
        requests.post(
            url, json={"count": len(rows), "rows": rows}, timeout=10
        ).raise_for_status()
    return {"sent": len(rows)}


@app.local_entrypoint()
def main(batch: int = 100, timeout_s: int = 5, local: bool = False):
    fn = etl.local if local else etl.remote
    print(fn(batch=batch, timeout_s=timeout_s))

