"""Sarvam LLM service wrapper for Pipecat pipelines.

This helper keeps the bot entrypoint clean while configuring Pipecat's
OpenAI-compatible client with Sarvam defaults.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService

SARVAM_DEFAULT_MODEL = "sarvam-m"
SARVAM_DEFAULT_BASE_URL = "https://api.sarvam.ai/v1"


class SarvamLLMService(OpenAILLMService):
    """Pipecat LLM service configured for Sarvam's OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        params: Optional[BaseOpenAILLMService.InputParams] = None,
        **kwargs,
    ) -> None:
        resolved_api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Sarvam API key not provided. Pass `api_key` or set the `SARVAM_API_KEY` env var."
            )

        resolved_model = model or os.getenv("SARVAM_LLM_MODEL", SARVAM_DEFAULT_MODEL)
        resolved_base_url = base_url or os.getenv(
            "SARVAM_API_BASE_URL", SARVAM_DEFAULT_BASE_URL
        )

        super().__init__(
            model=resolved_model,
            api_key=resolved_api_key,
            base_url=resolved_base_url.rstrip("/"),
            organization=organization,
            project=project,
            default_headers=default_headers,
            params=params,
            **kwargs,
        )


__all__ = ["SarvamLLMService"]
