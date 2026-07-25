# ---
# lambda-test: false  # missing-secret
# ---

# # Export Modal telemetry to Parseable OSS
#
# The official OpenTelemetry Python SDK exports OTLP/HTTP as protobuf, while
# Parseable OSS ingests OTLP/JSON. This example places an OpenTelemetry Collector
# between Modal and Parseable to convert all three signals from protobuf to JSON.
#
# ```text
# Modal Function -- OTLP/protobuf --> Collector -- OTLP/JSON --> Parseable OSS
# ```

import logging
import os
import time

import modal

otel_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "opentelemetry-api==1.44.0",
    "opentelemetry-sdk==1.44.0",
    "opentelemetry-exporter-otlp-proto-http==1.44.0",
    "opentelemetry-instrumentation-logging==0.65b0",
)

app = modal.App("example-parseable-otel-oss")

with otel_image.imports():
    from opentelemetry import metrics, trace
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.logging.handler import LoggingHandler
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

# ## Configure the Collector
#
# Copy `.env.oss.example` to `.env.oss`, replace its placeholders, and start the
# adjacent Collector. `PARSEABLE_ENDPOINT` is the Parseable OSS base URL without
# a signal path. OSS is single-tenant, so this path does not send `X-P-Tenant`.
#
# ```shell
# cd 10_integrations/parseable
# cp .env.oss.example .env.oss
# docker compose -f compose.oss.yaml up -d
# cloudflared tunnel --url http://localhost:4318
# ```
#
# Create a Modal Secret using the public HTTPS URL for that Collector and the
# same ingress token configured in `.env.oss`:
#
# ```shell
# modal secret create parseable-otel-oss \
#   OTEL_COLLECTOR_ENDPOINT="https://collector.example.com" \
#   OTEL_COLLECTOR_TOKEN="replace-me"
# ```

otel_secret = modal.Secret.from_name(
    "parseable-otel-oss",
    required_keys=["OTEL_COLLECTOR_ENDPOINT", "OTEL_COLLECTOR_TOKEN"],
)


@app.cls(image=otel_image, secrets=[otel_secret])
class InstrumentedWorker:
    @modal.enter()
    def setup_telemetry(self):
        resource = Resource.create(
            {
                "service.name": "modal-parseable-oss-example",
                "service.namespace": "modal-examples",
                "deployment.environment.name": "demo",
            }
        )

        endpoint = os.environ["OTEL_COLLECTOR_ENDPOINT"].rstrip("/")
        headers = {"Authorization": f"Bearer {os.environ['OTEL_COLLECTOR_TOKEN']}"}
        span_exporter = OTLPSpanExporter(
            endpoint=f"{endpoint}/v1/traces", headers=headers
        )
        log_exporter = OTLPLogExporter(endpoint=f"{endpoint}/v1/logs", headers=headers)
        metric_exporter = OTLPMetricExporter(
            endpoint=f"{endpoint}/v1/metrics", headers=headers
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

        self.tracer = trace.get_tracer("modal.parseable.oss.example")
        meter = metrics.get_meter("modal.parseable.oss.example")
        self.invocations = meter.create_counter(
            "demo.invocations", description="Number of example calls"
        )
        self.duration = meter.create_histogram(
            "demo.work.duration", unit="ms", description="Example work duration"
        )

        self.logger = logging.getLogger("modal.parseable.oss.example")
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

        self.tracer_provider.force_flush(timeout_millis=5_000)
        self.logger_provider.force_flush(timeout_millis=5_000)
        self.meter_provider.force_flush(timeout_millis=5_000)

        return (
            "Telemetry exported through the Collector. In Parseable, inspect "
            "modal-logs, modal-traces, and modal-metrics."
        )

    @modal.exit()
    def shutdown_telemetry(self):
        self.tracer_provider.force_flush(timeout_millis=5_000)
        self.logger_provider.force_flush(timeout_millis=5_000)
        self.meter_provider.force_flush(timeout_millis=5_000)
        self.logger_provider.shutdown()
        self.meter_provider.shutdown()
        self.tracer_provider.shutdown()


# ## Run the example
#
# ```shell
# modal run 10_integrations/parseable/parseable_otel_oss.py
# ```


@app.local_entrypoint()
def main(name: str = "Modal"):
    print(InstrumentedWorker().run.remote(name))
