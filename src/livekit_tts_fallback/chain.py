from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from livekit.agents import tts

from .errors import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FallbackPolicy:
    """Configuration passed to LiveKit's native TTS fallback adapter."""

    max_retry_per_tts: int = 0
    output_sample_rate: int | None = 24_000
    prewarm_fallbacks: bool = False

    def __post_init__(self) -> None:
        if self.max_retry_per_tts < 0:
            raise ConfigurationError("max_retry_per_tts cannot be negative")
        if self.output_sample_rate is not None and self.output_sample_rate <= 0:
            raise ConfigurationError("output_sample_rate must be positive")


class ManagedFallbackAdapter(tts.FallbackAdapter):
    """LiveKit FallbackAdapter with explicit provider ownership and lifecycle."""

    def __init__(
        self,
        providers: Sequence[tts.TTS],
        *,
        policy: FallbackPolicy | None = None,
        close_providers: bool = True,
    ) -> None:
        if not providers:
            raise ConfigurationError("at least one TTS provider is required")
        if len({id(provider) for provider in providers}) != len(providers):
            raise ConfigurationError("the same TTS provider instance cannot appear twice")

        self._managed_providers = tuple(providers)
        self._policy = policy or FallbackPolicy()
        self._close_providers = close_providers
        self._closed = False

        super().__init__(
            list(self._managed_providers),
            max_retry_per_tts=self._policy.max_retry_per_tts,
            sample_rate=self._policy.output_sample_rate,
        )
        self.on("tts_availability_changed", self._log_availability_change)

    @property
    def providers(self) -> tuple[tts.TTS, ...]:
        return self._managed_providers

    def prewarm(self) -> None:
        targets = (
            self._managed_providers
            if self._policy.prewarm_fallbacks
            else self._managed_providers[:1]
        )
        for provider in targets:
            provider.prewarm()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.off("tts_availability_changed", self._log_availability_change)
        await super().aclose()

        if not self._close_providers:
            return

        results = await asyncio.gather(
            *(provider.aclose() for provider in self._managed_providers),
            return_exceptions=True,
        )
        for provider, result in zip(self._managed_providers, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "failed to close TTS provider",
                    extra={"provider": provider.label, "error": repr(result)},
                )

    @staticmethod
    def _log_availability_change(event: object) -> None:
        provider = getattr(event, "tts", None)
        logger.info(
            "TTS provider availability changed",
            extra={
                "provider": getattr(provider, "provider", "unknown"),
                "label": getattr(provider, "label", "unknown"),
                "available": getattr(event, "available", None),
            },
        )


def build_fallback_tts(
    primary: tts.TTS,
    *,
    fallbacks: Sequence[tts.TTS] = (),
    policy: FallbackPolicy | None = None,
    close_providers: bool = True,
) -> ManagedFallbackAdapter:
    """Build an ordered LiveKit fallback chain from any LiveKit TTS providers."""

    return ManagedFallbackAdapter(
        (primary, *fallbacks),
        policy=policy,
        close_providers=close_providers,
    )
