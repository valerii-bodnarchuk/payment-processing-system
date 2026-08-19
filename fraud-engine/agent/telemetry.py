"""
OpenTelemetry bootstrap — SDK wiring and nothing else.

This module only stands the tracing pipeline up. It deliberately instruments
nothing: graph nodes and LLM calls stay untouched and get their spans in later
changes. What lands here is the contract those changes build on — a tracer that
is safe to import at module scope, and a trace id that is safe to log.

Failure policy: tracing is observability, not a dependency. An unreachable
collector must degrade to dropped spans, never to a failed request. The OTLP
exporter runs inside BatchSpanProcessor's worker thread, so export errors are
logged there and never propagate into the request path.
"""
from __future__ import annotations

import logging
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger("agent.telemetry")

DEFAULT_SERVICE_NAME = "fraud-engine"
DEFAULT_OTLP_ENDPOINT = "http://localhost:4317"

# OTEL_SDK_DISABLED is specified as a boolean, but operators reach for more than
# "true" — accept the usual spellings rather than silently tracing because
# someone wrote "1".
_TRUTHY = frozenset({"true", "1", "yes", "on"})

_provider: TracerProvider | None = None
_initialized = False

# Acquired at import time, before any provider exists. The API hands back a
# ProxyTracer that resolves to the global provider on first use, so importers do
# not have to care whether setup_telemetry() has run yet — `from agent.telemetry
# import tracer` at module scope is safe anywhere, including in nodes that are
# imported long before the app starts.
tracer = trace.get_tracer("fraud-engine.agent")


def _is_disabled() -> bool:
    return os.getenv("OTEL_SDK_DISABLED", "").strip().lower() in _TRUTHY


def setup_telemetry() -> None:
    """Register the global TracerProvider. Idempotent — a second call is a no-op.

    With OTEL_SDK_DISABLED set, no provider is registered at all: the API's
    default no-op provider stays in place, so `tracer` keeps working and every
    span it creates is dropped without touching the network. The disabled
    decision is made once per process, like the enabled one.
    """
    global _provider, _initialized

    if _initialized:
        return

    # Nothing else in this process has loaded .env: app/main.py never imports
    # agent.config, and the OTEL_* values live in the repo-root file. Guarded
    # because a missing env_bootstrap should fall back to the process
    # environment, not switch tracing off.
    try:
        from env_bootstrap import load_env

        load_env()
    except ImportError:
        logger.debug("env_bootstrap unavailable — reading OTEL_* from the process env only")

    if _is_disabled():
        _initialized = True
        logger.info("OTEL_SDK_DISABLED is set — tracing stays on the no-op provider")
        return

    service_name = os.getenv("OTEL_SERVICE_NAME") or DEFAULT_SERVICE_NAME
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or DEFAULT_OTLP_ENDPOINT

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    # An http:// endpoint selects an insecure gRPC channel on its own; passing
    # `insecure` here would override OTEL_EXPORTER_OTLP_INSECURE for operators
    # who set it deliberately.
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    )
    trace.set_tracer_provider(provider)

    _provider = provider
    _initialized = True
    logger.info(
        "Tracing initialised: service.name=%s, OTLP endpoint=%s",
        service_name,
        endpoint,
    )


def shutdown_telemetry() -> None:
    """Flush buffered spans and stop the exporter.

    Wraps TracerProvider.shutdown(), which force-flushes the BatchSpanProcessor
    before stopping its worker — without it the last batch dies with the
    process. Safe to call when setup never ran or was disabled.
    """
    global _provider, _initialized

    provider, _provider = _provider, None
    _initialized = False

    if provider is not None:
        provider.shutdown()


def current_trace_id() -> str | None:
    """Trace id of the active span as 32-char hex, or None when there is no span.

    Returns None rather than a zeroed id for the no-span case: a log line
    correlating to trace 000…0 is worse than one that admits it has no trace,
    because the former looks joinable and silently is not. This is also what a
    no-op provider produces, so tracing being disabled reads the same as tracing
    not having started.
    """
    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return None
    return trace.format_trace_id(span_context.trace_id)
