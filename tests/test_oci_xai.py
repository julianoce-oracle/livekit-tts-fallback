from __future__ import annotations

from collections.abc import AsyncIterator
from urllib.parse import parse_qs, urlparse

import pytest

from livekit_tts_fallback.errors import ConfigurationError
from livekit_tts_fallback.providers.oci_xai import OciXaiConfig, OciXaiTTS
from livekit_tts_fallback.transports import ConnectionLease


class FakeXaiConnection:
    def __init__(self) -> None:
        self.sent_text: list[str] = []
        self.finish_calls = 0
        self.closed = False

    @property
    def is_open(self) -> bool:
        return not self.closed

    async def close(self) -> None:
        self.closed = True

    async def send_text(self, text: str) -> None:
        self.sent_text.append(text)

    async def finish_text(self) -> None:
        self.finish_calls += 1

    async def receive_audio(self) -> AsyncIterator[bytes]:
        yield b"\x01\x00" * 4_800


class FakePool:
    def __init__(self, connection: FakeXaiConnection) -> None:
        self.connection = connection
        self.releases: list[bool] = []

    async def start(self) -> None:
        return None

    async def acquire(self, *, timeout_s: float | None = None) -> ConnectionLease:
        return ConnectionLease(entry_id="one", connection=self.connection, reused=True)

    async def release(self, lease: ConnectionLease, *, healthy: bool) -> None:
        self.releases.append(healthy)

    async def aclose(self) -> None:
        await self.connection.close()

    def snapshot(self) -> dict[str, int | bool]:
        return {"size": 1, "idle": 1, "leased": 0, "closed": False}


def test_oci_xai_omits_optional_query_parameters_by_default() -> None:
    query = parse_qs(urlparse(OciXaiConfig(api_key="test-key").websocket_url()).query)

    assert "optimize_streaming_latency" not in query
    assert "text_normalization" not in query


def test_oci_xai_encodes_optional_query_parameters_when_configured() -> None:
    config = OciXaiConfig(
        api_key="test-key",
        optimize_streaming_latency=1,
        text_normalization=True,
    )
    query = parse_qs(urlparse(config.websocket_url()).query)

    assert query["optimize_streaming_latency"] == ["1"]
    assert query["text_normalization"] == ["true"]


def test_oci_xai_rejects_invalid_streaming_latency() -> None:
    with pytest.raises(ConfigurationError, match="optimize_streaming_latency"):
        OciXaiConfig(api_key="test-key", optimize_streaming_latency=3)


@pytest.mark.asyncio
async def test_oci_xai_streams_pcm_through_livekit() -> None:
    connection = FakeXaiConnection()
    pool = FakePool(connection)
    provider = OciXaiTTS(
        OciXaiConfig(api_key="test-key"),  # type: ignore[arg-type]
        _pool=pool,  # type: ignore[arg-type]
    )

    async with provider.synthesize("ola") as stream:
        events = [event async for event in stream]

    assert events
    assert connection.sent_text == ["ola"]
    assert connection.finish_calls == 1
    assert pool.releases == [True]
    await provider.aclose()
