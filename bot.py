import os
from typing import Optional, Type, TYPE_CHECKING

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.exotel import ExotelFrameSerializer
from pipecat.services.sarvam.tts import SarvamTTSService
from sarvam_llm import SarvamLLMService
from sarvam_stt import SarvamSTTService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv(override=True)

base_dir = os.path.dirname(__file__)
prompt_path = os.path.join(base_dir, "prompts", "system_prompt.md")
with open(prompt_path, "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()

if TYPE_CHECKING:
    from pipecat.processors.metrics.sentry import SentryMetrics

_SENTRY_SETUP_ATTEMPTED = False
_SENTRY_METRICS_CLASS: Optional[Type["SentryMetrics"]] = None


def _is_sentry_enabled() -> bool:
    raw_value = os.getenv("SENTRY_METRICS_ENABLED")
    if raw_value is None:
        return True
    normalized = raw_value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


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


def _create_sentry_metrics_instance() -> Optional["SentryMetrics"]:
    metrics_class = _ensure_sentry_metrics_class()
    if not metrics_class:
        return None
    return metrics_class()


async def run_bot(transport: BaseTransport, handle_sigint: bool):
    llm_metrics = _create_sentry_metrics_instance()
    stt_metrics = _create_sentry_metrics_instance()
    tts_metrics = _create_sentry_metrics_instance()

    llm_kwargs = {"metrics": llm_metrics} if llm_metrics else {}
    stt_kwargs = {"metrics": stt_metrics} if stt_metrics else {}
    tts_kwargs = {"metrics": tts_metrics} if tts_metrics else {}

    llm = SarvamLLMService(**llm_kwargs)

    stt = SarvamSTTService(**stt_kwargs)

    tts = SarvamTTSService(
        api_key=os.getenv("SARVAM_API_KEY"),
        model="bulbul:v2",
        voice_id="anushka",
        **tts_kwargs,
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Starting outbound call conversation")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""

    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Auto-detected transport: {transport_type}")

    serializer = ExotelFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
    )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    handle_sigint = runner_args.handle_sigint

    await run_bot(transport, handle_sigint)
