# ---
# lambda-test: false  # missing-secret
# ---

# # Export Modal telemetry to Parseable with OpenTelemetry
#
# This example sends application logs, traces, and metrics from a Modal Function
# to [Parseable](https://www.parseable.com/) with the OpenTelemetry Python SDK.
# Logs contain their current trace and span IDs, so you can jump from a log record
# to the span that produced it.
#
# ```text
# Modal Function -- OTLP/HTTP --> Parseable
# ```

import logging
import os
import time

import modal

otel_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "opentelemetry-api==1.44.0",
    "opentelemetry-sdk==1.44.0",
    "opentelemetry-exporter-otlp-proto-http==1.44.0",
)

app = modal.App("example-parseable-otel")

with otel_image.imports():
    from opentelemetry import metrics, trace
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

# ## Configure the Parseable destination
#
# Create a Modal Secret named `parseable-otel` with your Parseable base endpoint,
# an API key with the `ingestor` role, and tenant ID. See the
# [OpenTelemetry](https://www.parseable.com/docs/ingest-data/otel) and
# [API Keys](https://www.parseable.com/docs/user-guide/api-keys) docs. In the
# Parseable UI, generate the endpoint via Getting Started -> Ingest
# telemetry data -> OTel -> Set up -> Generate.
#
# Do not include `v1/logs`, `v1/traces`, or `v1/metrics` as each exporter appends
# its signal path. The tenant ID is the `workspaceId` in the Parseable app URL.
#
# ```shell
# modal secret create parseable-otel \
#   PARSEABLE_ENDPOINT="https://parseable.example.com" \
#   PARSEABLE_API_KEY="replace-me" \
#   PARSEABLE_TENANT_ID="replace-me"
# ```

otel_secret = modal.Secret.from_name(
    "parseable-otel",
    required_keys=[
        "PARSEABLE_ENDPOINT",
        "PARSEABLE_API_KEY",
        "PARSEABLE_TENANT_ID",
    ],
)


# ## Create the exporter worker
#
# A [modal.Cls](https://modal.com/docs/sdk/py/latest/Cls) initializes the
# exporters once per container and shuts them down when the container exits.


def _parseable_headers(stream: str, log_source: str) -> dict[str, str]:
    return {
        "X-API-Key": os.environ["PARSEABLE_API_KEY"],
        "X-P-Tenant": os.environ["PARSEABLE_TENANT_ID"],
        "X-P-Stream": stream,
        "X-P-Log-Source": log_source,
    }


@app.cls(image=otel_image, secrets=[otel_secret])
class InstrumentedWorker:
    @modal.enter()
    def setup_telemetry(self):
        resource = Resource.create(
            {
                "service.name": "modal-parseable-example",
                "service.namespace": "modal-examples",
                "deployment.environment.name": "demo",
            }
        )

        endpoint = os.environ["PARSEABLE_ENDPOINT"].rstrip("/")
        span_exporter = OTLPSpanExporter(
            endpoint=f"{endpoint}/v1/traces",
            headers=_parseable_headers("modal-traces", "otel-traces"),
        )
        log_exporter = OTLPLogExporter(
            endpoint=f"{endpoint}/v1/logs",
            headers=_parseable_headers("modal-logs", "otel-logs"),
        )
        metric_exporter = OTLPMetricExporter(
            endpoint=f"{endpoint}/v1/metrics",
            headers=_parseable_headers("modal-metrics", "otel-metrics"),
        )

        self.tracer_provider = TracerProvider(resource=resource)
        self.tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(self.tracer_provider)

        self.logger_provider = LoggerProvider(resource=resource)
        self.logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(log_exporter)
        )
        set_logger_provider(self.logger_provider)

        metric_reader = PeriodicExportingMetricReader(
            metric_exporter, export_interval_millis=5_000
        )
        self.meter_provider = MeterProvider(
            resource=resource, metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(self.meter_provider)

        self.tracer = trace.get_tracer("modal.parseable.example")
        meter = metrics.get_meter("modal.parseable.example")
        self.invocations = meter.create_counter(
            "demo.invocations", description="Number of example calls"
        )
        self.duration = meter.create_histogram(
            "demo.work.duration", unit="ms", description="Example work duration"
        )

        self.logger = logging.getLogger("modal.parseable.example")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(
            LoggingHandler(level=logging.INFO, logger_provider=self.logger_provider)
        )
        self.logger.propagate = False

    @modal.method()
    def run(self, name: str = "Modal") -> str:
        started_at = time.perf_counter()

        with self.tracer.start_as_current_span("demo.run") as span:
            span.set_attribute("demo.name", name)
            span.set_attribute("modal.function.kind", "class_method")
            self.logger.info("Starting example work for %s", name)

            with self.tracer.start_as_current_span("demo.simulated_work"):
                time.sleep(0.1)

            elapsed_ms = (time.perf_counter() - started_at) * 1_000
            self.invocations.add(1, {"result": "success"})
            self.duration.record(elapsed_ms, {"operation": "demo.run"})
            self.logger.info("Finished example work in %.2f ms", elapsed_ms)

        # Batch processors export asynchronously; flush so a short run still
        # exercises the full OTLP path before the method returns.
        self.tracer_provider.force_flush(timeout_millis=5_000)
        self.logger_provider.force_flush(timeout_millis=5_000)
        self.meter_provider.force_flush(timeout_millis=5_000)

        return (
            "Telemetry exported. Confirm the demo signals in Parseable:\n"
            "1. Open https://app.parseable.com/\n"
            "2. Select your workspace\n"
            "3. In modal-logs: look for 'Starting example work' and "
            "'Finished example work'\n"
            "4. In modal-traces: look for demo.run and demo.simulated_work\n"
            "5. In modal-metrics: look for demo.invocations and "
            "demo.work.duration"
        )

    @modal.exit()
    def shutdown_telemetry(self):
        self.tracer_provider.force_flush(timeout_millis=5_000)
        self.logger_provider.force_flush(timeout_millis=5_000)
        self.meter_provider.force_flush(timeout_millis=5_000)
        self.logger_provider.shutdown()
        self.meter_provider.shutdown()
        self.tracer_provider.shutdown()


# ## Run and inspect the signals
#
# Create the Secret above, then run:
#
# ```shell
# modal run 10_integrations/parseable/parseable_otel.py
# ```
#
# Parseable will contain spans named `demo.run` and `demo.simulated_work`, logs
# beginning `Starting example work` and `Finished example work`, and metrics named
# `demo.invocations` and `demo.work.duration`. Application log records emitted
# inside `demo.run` also carry its trace and span IDs.


@app.local_entrypoint()
def main(name: str = "Modal"):
    print(InstrumentedWorker().run.remote(name))
