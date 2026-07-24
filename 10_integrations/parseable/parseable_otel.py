# ---
# lambda-test: false  # missing-secret
# ---

# # Export Modal telemetry to Parseable with OpenTelemetry
#
# This example sends application logs, traces, and metrics from a Modal Function
# to [Parseable](https://www.parseable.com/) through an OpenTelemetry Collector.
# The three signals share one resource and trace context, making it possible to
# move from a log record to the span that produced it.
#
# The Collector runs outside Modal:
#
# ```text
# Modal Function -- OTLP/HTTP --> OpenTelemetry Collector --> Parseable
# ```
#
# A separately hosted Collector gives every Modal container one durable gateway
# and avoids producing telemetry about the telemetry gateway itself. For a local
# demonstration, the files next to this example run the Collector with Docker
# Compose. A temporary HTTPS tunnel makes it reachable from Modal.

import logging
import time

import modal

# ## Build an Image with OpenTelemetry
#
# Only `modal` is needed locally. OpenTelemetry packages are installed in the
# remote [Image](https://modal.com/docs/guide/images).

otel_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "opentelemetry-api==1.36.0",
    "opentelemetry-sdk==1.36.0",
    "opentelemetry-exporter-otlp-proto-http==1.36.0",
)

app = modal.App("example-parseable-otel")

# ## Configure the OTLP destination
#
# Create a Modal Secret named `parseable-otel` containing these standard
# OpenTelemetry environment variables:
#
# ```shell
# modal secret create parseable-otel \
#   OTEL_EXPORTER_OTLP_ENDPOINT="https://collector.example.com" \
#   OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf" \
#   OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer replace-me" \
#   OTEL_HEADER_Authorization="Bearer replace-me"
# ```
#
# The endpoint is the Collector, not Parseable. A Parseable API key with the
# `ingestor` role remains in the Collector environment and is never copied into
# Modal.
#
# For the local Collector included with this example, copy `.env.example` to
# `.env`, replace its placeholders, and start it:
#
# ```shell
# cd 10_integrations/parseable
# docker compose up -d
# cloudflared tunnel --url http://localhost:4318
# ```
#
# A quick tunnel is useful for testing, but its URL changes when restarted. Use
# a stable, authenticated HTTPS endpoint for production.

otel_secret = modal.Secret.from_name(
    "parseable-otel",
    required_keys=[
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_HEADER_Authorization",
    ],
)


# ## Instrument a Modal Cls
#
# A [Cls](https://modal.com/docs/guide/lifecycle-functions) lets us initialize
# providers once per container, then flush them when that container exits.


@app.cls(image=otel_image, secrets=[otel_secret])
class InstrumentedWorker:
    @modal.enter()
    def setup_telemetry(self):
        from opentelemetry import metrics, trace
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.http._log_exporter import (
            OTLPLogExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {
                "service.name": "modal-parseable-example",
                "service.namespace": "modal-examples",
                "deployment.environment.name": "demo",
            }
        )

        self.tracer_provider = TracerProvider(resource=resource)
        self.tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(self.tracer_provider)

        self.logger_provider = LoggerProvider(resource=resource)
        self.logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter())
        )
        set_logger_provider(self.logger_provider)

        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(), export_interval_millis=5_000
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
    def run(self, name: str = "Modal") -> dict[str, str]:
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

        return {"message": f"Hello, {name}!", "telemetry": "exported"}

    @modal.exit()
    def shutdown_telemetry(self):
        # Batch processors export asynchronously. Flush them so short-lived
        # containers do not lose their final records during a normal shutdown.
        self.tracer_provider.force_flush(timeout_millis=5_000)
        self.logger_provider.force_flush(timeout_millis=5_000)
        self.meter_provider.force_flush(timeout_millis=5_000)
        self.logger_provider.shutdown()
        self.meter_provider.shutdown()
        self.tracer_provider.shutdown()


# ## Run and inspect the signals
#
# Start the Collector using the adjacent `compose.yaml`, expose port 4318 through
# a stable HTTPS endpoint or temporary tunnel, create the Secret above, then run:
#
# ```shell
# modal run 10_integrations/parseable/parseable_otel.py
# ```
#
# Parseable will contain spans named `demo.run` and `demo.simulated_work`, logs
# beginning `Starting example work` and `Finished example work`, and metrics named
# `demo.invocations` and `demo.work.duration`. Application log records emitted
# inside `demo.run` also carry its trace and span IDs.
#
# Modal can send platform logs and container metrics through the same Collector.
# In **Workspace Settings -> OpenTelemetry**, enter the Collector base URL, select
# the `parseable-otel` Secret, then test and save the configuration. The
# `OTEL_HEADER_Authorization` key above authenticates this managed export.


@app.local_entrypoint()
def main(name: str = "Modal"):
    print(InstrumentedWorker().run.remote(name))
