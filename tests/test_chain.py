from __future__ import annotations

import asyncio

import pytest
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    tts,
    utils,
)

from livekit_tts_fallback import FallbackPolicy, build_fallback_tts


class FakeTTS(tts.TTS):
    def __init__(self, *, payload: bytes, failure: str | None = None) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24_000,
            num_channels=1,
        )
        self.payload = payload
        self.failure = failure
        self.calls = 0
        self.closed = False

    @property
    def provider(self) -> str:
        return "fake"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        self.calls += 1
        return FakeChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        self.closed = True


class FakeChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        provider = self._tts
        assert isinstance(provider, FakeTTS)
        if provider.failure == "before":
            raise APIConnectionError("failure before audio")

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=provider.sample_rate,
            num_channels=provider.num_channels,
            mime_type="audio/pcm",
        )
        output_emitter.push(provider.payload)
        output_emitter.flush()

        if provider.failure == "after":
            await asyncio.sleep(0.05)
            raise APIConnectionError("failure after audio")


@pytest.mark.asyncio
async def test_falls_back_to_any_livekit_tts_provider() -> None:
    primary = FakeTTS(payload=b"\x00\x00" * 4_800, failure="before")
    fallback = FakeTTS(payload=b"\x01\x00" * 4_800)
    chain = build_fallback_tts(
        primary,
        fallbacks=[fallback],
        policy=FallbackPolicy(max_retry_per_tts=0),
    )

    events = []
    async with chain.synthesize(
        "hello",
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
    ) as stream:
        events = [event async for event in stream]

    assert events
    assert primary.calls >= 1
    assert fallback.calls == 1
    await chain.aclose()
    assert primary.closed is True
    assert fallback.closed is True


@pytest.mark.asyncio
async def test_does_not_start_fallback_after_partial_audio() -> None:
    primary = FakeTTS(payload=b"\x01\x00" * 4_800, failure="after")
    fallback = FakeTTS(payload=b"\x02\x00" * 4_800)
    chain = build_fallback_tts(primary, fallbacks=[fallback])

    async with chain.synthesize("hello") as stream:
        events = [event async for event in stream]

    assert events
    assert fallback.calls == 0
    await chain.aclose()


def test_rejects_the_same_provider_instance_twice() -> None:
    provider = FakeTTS(payload=b"\x00\x00" * 4_800)
    with pytest.raises(ValueError, match="cannot appear twice"):
        build_fallback_tts(provider, fallbacks=[provider])
