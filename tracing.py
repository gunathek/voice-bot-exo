"""Helpers for configuring Sentry metrics and Langfuse tracing."""

from __future__ import annotations

import os
from typing import Optional, Type, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - import guard
    from pipecat.processors.metrics.sentry import SentryMetrics


_SENTRY_SETUP_ATTEMPTED = False
_SENTRY_METRICS_CLASS: Optional[Type["SentryMetrics"]] = None


def env_flag_enabled(env_name: str, *, default: bool = False) -> bool:
    """Return True when the environment variable is set to a truthy value."""
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if not normalized:
        return default

    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _is_sentry_enabled() -> bool:
    return env_flag_enabled("SENTRY_METRICS_ENABLED", default=True)


def _parse_sample_rate(env_name: str, default: Optional[float] = None) -> Optional[float]:
    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        logger.warning(
            f"Invalid value '{raw_value}' for {env_name}; falling back to {default!r}."
        )
        return default


def _ensure_sentry_metrics_class() -> Optional[Type["SentryMetrics"]]:
    global _SENTRY_SETUP_ATTEMPTED
    global _SENTRY_METRICS_CLASS

    if _SENTRY_METRICS_CLASS or _SENTRY_SETUP_ATTEMPTED:
        return _SENTRY_METRICS_CLASS

    _SENTRY_SETUP_ATTEMPTED = True

    if not _is_sentry_enabled():
        logger.info("Sentry metrics disabled via SENTRY_METRICS_ENABLED")
        return None

    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        return None

    try:  # Import lazily so we don't require sentry unless configured
        import sentry_sdk
        from pipecat.processors.metrics.sentry import SentryMetrics
    except ImportError as exc:  # pragma: no cover - runtime guard
        logger.warning(
            f"SENTRY_DSN is set, but required dependencies are missing: {exc}. "
            "Install 'pipecat-ai[sentry]' to enable Sentry metrics."
        )
        return None

    traces_sample_rate = _parse_sample_rate("SENTRY_TRACES_SAMPLE_RATE", 1.0)
    profiles_sample_rate = _parse_sample_rate("SENTRY_PROFILES_SAMPLE_RATE")
    environment = os.getenv("SENTRY_ENVIRONMENT", "").strip() or None
    release = os.getenv("SENTRY_RELEASE", "").strip() or None

    sentry_options: dict[str, object] = {
        "dsn": dsn,
        "traces_sample_rate": traces_sample_rate,
    }
    if profiles_sample_rate is not None:
        sentry_options["profiles_sample_rate"] = profiles_sample_rate
    if environment:
        sentry_options["environment"] = environment
    if release:
        sentry_options["release"] = release

    sentry_sdk.init(**sentry_options)
    logger.info("Sentry metrics enabled for Pipecat services")

    _SENTRY_METRICS_CLASS = SentryMetrics
    return _SENTRY_METRICS_CLASS


def create_sentry_metrics_instance() -> Optional["SentryMetrics"]:
    metrics_class = _ensure_sentry_metrics_class()
    if not metrics_class:
        return None
    return metrics_class()


def _initialize_tracing() -> bool:
    if not env_flag_enabled("ENABLE_TRACING"):
        return False

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from pipecat.utils.tracing.setup import setup_tracing

        otlp_exporter = OTLPSpanExporter()
        setup_tracing(
            service_name=os.getenv("OTEL_SERVICE_NAME", "voice-bot-pipecat"),
            exporter=otlp_exporter,
            console_export=env_flag_enabled("OTEL_CONSOLE_EXPORT"),
        )
        logger.info("OpenTelemetry tracing enabled for Langfuse exporter")
        return True
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
        logger.warning(
            "ENABLE_TRACING is set but OpenTelemetry extras are missing: {}. Run `uv sync` to "
            "install updated dependencies.",
            exc,
        )
        return False
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.warning(f"Failed to initialize OpenTelemetry tracing: {exc}")
        return False


IS_TRACING_ENABLED = _initialize_tracing()


__all__ = [
    "create_sentry_metrics_instance",
    "env_flag_enabled",
    "IS_TRACING_ENABLED",
]
