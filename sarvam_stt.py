"""Sarvam speech-to-text integration for Pipecat pipelines.

This module provides a segmented (chunked) STT service that reuses the
pipeline's existing VAD to detect speech boundaries and submits complete
segments to Sarvam's HTTP transcription endpoint.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncGenerator, Mapping, Optional

from loguru import logger

from pipecat.frames.frames import CancelFrame, EndFrame, ErrorFrame, Frame, StartFrame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import aiohttp
except ModuleNotFoundError as exc:  # pragma: no cover - handled at import time
    logger.error(f"Exception: {exc}")
    logger.error("In order to use Sarvam STT, you need to `pip install pipecat-ai[sarvam]`.")
    raise


SARVAM_STT_DEFAULT_MODEL = "saarika:v2.5"
SARVAM_STT_DEFAULT_LANGUAGE = "en-IN"
SARVAM_STT_DEFAULT_BASE_URL = "https://api.sarvam.ai/speech-to-text"


def _load_extra_params_from_env() -> dict[str, Any]:
    raw = os.getenv("SARVAM_STT_EXTRA_PARAMS_JSON", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse SARVAM_STT_EXTRA_PARAMS_JSON; ignoring value.")
        return {}
    if isinstance(parsed, Mapping):
        return dict(parsed)
    logger.warning("SARVAM_STT_EXTRA_PARAMS_JSON must be a JSON object; ignoring value.")
    return {}


def _normalize_language(language: Optional[str | Language]) -> str:
    if not language:
        return SARVAM_STT_DEFAULT_LANGUAGE
    if isinstance(language, Language):
        return language.value
    return str(language)


class SarvamSTTService(SegmentedSTTService):
    """Segmented Sarvam speech-to-text service for Pipecat.

    The service buffers audio between VAD start/stop events and POSTs each
    segment to Sarvam's `/speech-to-text` endpoint. This keeps the pipeline's
    own stereo VAD in control while still taking advantage of Sarvam's models.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        language: Optional[str | Language] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        extra_params: Optional[Mapping[str, Any]] = None,
        request_timeout: float = 15.0,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
        sample_rate: Optional[int] = None,
        **kwargs,
    ) -> None:
        resolved_api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Sarvam API key not provided. Pass `api_key` or set the `SARVAM_API_KEY` env var."
            )

        resolved_model = model or os.getenv("SARVAM_STT_MODEL", SARVAM_STT_DEFAULT_MODEL)
        resolved_language = _normalize_language(
            language or os.getenv("SARVAM_STT_LANGUAGE", SARVAM_STT_DEFAULT_LANGUAGE)
        )
        resolved_base_url = (
            base_url or os.getenv("SARVAM_STT_BASE_URL", SARVAM_STT_DEFAULT_BASE_URL)
        ).rstrip("/")
        resolved_extra_params = dict(extra_params or _load_extra_params_from_env())

        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = resolved_api_key
        self._base_url = resolved_base_url
        self._language_code = resolved_language
        self._extra_params = resolved_extra_params
        self._request_timeout = request_timeout
        self._session: Optional[aiohttp.ClientSession] = aiohttp_session
        self._own_session = False

        self.set_model_name(resolved_model)
        self._settings = {
            "language": self._language_code,
            "model": self.model_name,
        }

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # Ensure settings reflect negotiated sample rate.
        self._settings["sample_rate"] = self.sample_rate

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._close_session()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._close_session()

    async def set_model(self, model: str):
        await super().set_model(model)
        self._settings["model"] = model

    async def set_language(self, language: Language):
        code = _normalize_language(language)
        self._language_code = code
        self._settings["language"] = code

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not audio:
            logger.warning("Sarvam STT received empty audio segment; skipping request.")
            yield None
            return

        session = await self._ensure_session()
        form = aiohttp.FormData()
        form.add_field("file", audio, filename="segment.wav", content_type="audio/wav")
        form.add_field("language_code", self._language_code)
        form.add_field("model", self.model_name)

        for key, value in self._extra_params.items():
            if key in {"file", "language_code", "model"} or value is None:
                continue
            if isinstance(value, (dict, list)):
                form.add_field(key, json.dumps(value), content_type="application/json")
            elif isinstance(value, (bool, int, float)):
                form.add_field(key, json.dumps(value))
            else:
                form.add_field(key, str(value))

        headers = {"api-subscription-key": self._api_key}
        timeout = aiohttp.ClientTimeout(total=self._request_timeout)

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        try:
            async with session.post(
                self._base_url,
                data=form,
                headers=headers,
                timeout=timeout,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Sarvam STT error ({response.status}): {error_text}")
                    await self.stop_all_metrics()
                    yield ErrorFrame(f"Sarvam STT error: {error_text}")
                    return

                payload = await response.json()

            await self.stop_ttfb_metrics()

            transcript = str(payload.get("transcript", ""))
            language_code = payload.get("language_code") or self._language_code
            try:
                language = Language(language_code) if language_code else None
            except ValueError:
                language = None

            timestamp = time_now_iso8601()

            yield TranscriptionFrame(
                transcript,
                self._user_id,
                timestamp,
                language,
                result=payload,
            )

            await self.stop_processing_metrics()
        except asyncio.TimeoutError:
            logger.error("Sarvam STT request timed out")
            await self.stop_all_metrics()
            yield ErrorFrame("Sarvam STT request timed out")
        except aiohttp.ClientError as exc:
            logger.error(f"Sarvam STT connection error: {exc}")
            await self.stop_all_metrics()
            yield ErrorFrame(f"Sarvam STT connection error: {exc}")
        except Exception as exc:  # pragma: no cover - safeguard for unexpected issues
            logger.exception(f"Sarvam STT unexpected error: {exc}")
            await self.stop_all_metrics()
            yield ErrorFrame(f"Sarvam STT unexpected error: {exc}")

        yield None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session and not self._session.closed:
            return self._session
        self._session = aiohttp.ClientSession()
        self._own_session = True
        return self._session

    async def _close_session(self) -> None:
        if self._own_session and self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            self._own_session = False


__all__ = ["SarvamSTTService"]
