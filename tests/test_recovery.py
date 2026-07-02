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


class SequenceTTS(tts.TTS):
    def __init__(self, failures: list[bool]) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24_000,
            num_channels=1,
        )
        self.failures = failures
        self.calls = 0

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        call_index = self.calls
        self.calls += 1
        should_fail = call_index < len(self.failures) and self.failures[call_index]
        return SequenceStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            should_fail=should_fail,
        )

    async def aclose(self) -> None:
        return None


class SequenceStream(tts.ChunkedStream):
    def __init__(self, *, should_fail: bool, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.should_fail = should_fail

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        if self.should_fail:
            raise APIConnectionError("planned failure")
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=24_000,
            num_channels=1,
            mime_type="audio/pcm",
        )
        output_emitter.push(b"\x00\x00" * 4_800)
        output_emitter.flush()


@pytest.mark.asyncio
async def test_livekit_recovers_primary_for_the_next_request() -> None:
    primary = SequenceTTS([True, False, False])
    fallback = SequenceTTS([False])
    chain = build_fallback_tts(
        primary,
        fallbacks=[fallback],
        policy=FallbackPolicy(max_retry_per_tts=0),
    )
    recovered = asyncio.Event()

    def on_availability(event: object) -> None:
        if getattr(event, "tts", None) is primary and getattr(event, "available", False):
            recovered.set()

    chain.on("tts_availability_changed", on_availability)

    async with chain.synthesize("first") as stream:
        assert [event async for event in stream]
    await asyncio.wait_for(recovered.wait(), timeout=1.0)
    fallback_calls_after_first_request = fallback.calls

    async with chain.synthesize("second") as stream:
        assert [event async for event in stream]

    assert primary.calls >= 3
    assert fallback.calls == fallback_calls_after_first_request
    await chain.aclose()
