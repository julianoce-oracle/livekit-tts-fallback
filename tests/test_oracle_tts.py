from __future__ import annotations

import asyncio
import io
import threading
import wave
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import oci
import pytest
from livekit.agents import APIConnectOptions, APIStatusError

import oracle
import oracle.tts as oracle_tts


class FakeRawResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.stream_amounts: list[int] = []
        self.closed = False

    def stream(self, amount: int) -> Iterator[bytes]:
        self.stream_amounts.append(amount)
        yield from self.chunks

    def close(self) -> None:
        self.closed = True


class FakeStreamingData:
    def __init__(self, raw: FakeRawResponse) -> None:
        self.raw = raw
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeSpeechClient:
    def __init__(self, data: FakeStreamingData) -> None:
        self.data = data
        self.calls: list[Any] = []

    def synthesize_speech(self, details: object) -> object:
        self.calls.append(details)
        return SimpleNamespace(data=self.data)


def pcm_as_wav(pcm: bytes, *, sample_rate: int = 24_000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()


@pytest.mark.asyncio
async def test_oracle_tts_streams_pcm_and_closes_response() -> None:
    pcm = b"\x03\x00" * 4_800
    wav = pcm_as_wav(pcm)
    chunks = [wav[:2], wav[2:17], wav[17:43], wav[43:333], wav[333:]]

    raw = FakeRawResponse(chunks)
    data = FakeStreamingData(raw)
    client = FakeSpeechClient(data)
    provider = oracle.TTS(
        config={},
        region="us-chicago-1",
        compartment_id="ocid1.compartment.oc1..test",
        voice_id="test-voice",
        language_code="pt-BR",
        samples_rate_in_hz=24_000,
        chunk_size=777,
        _client_factory=lambda: client,
    )

    async with provider.synthesize(
        "ola",
        conn_options=APIConnectOptions(max_retry=0, timeout=2),
    ) as stream:
        events = [event async for event in stream]

    rendered_pcm = b"".join(event.frame.data.tobytes() for event in events)
    assert rendered_pcm == pcm
    assert raw.stream_amounts == [777]
    assert raw.closed is True
    assert data.closed is True

    details = client.calls[0]
    assert details.is_stream_enabled is True
    assert details.configuration.model_family == "ORACLE"
    assert details.configuration.model_details.model_name == "TTS_2_NATURAL"
    assert details.configuration.model_details.voice_id == "test-voice"
    assert details.configuration.model_details.language_code == "pt-BR"
    assert details.configuration.speech_settings.sample_rate_in_hz == 24_000
    assert details.configuration.speech_settings.output_format == "PCM"
    await provider.aclose()


class ErrorSpeechClient:
    def synthesize_speech(self, details: object) -> object:
        raise oci.exceptions.ServiceError(
            429,
            "TooManyRequests",
            {"opc-request-id": "request-123"},
            "rate limited",
        )


@pytest.mark.asyncio
async def test_oracle_tts_preserves_oci_status_error() -> None:
    provider = oracle.TTS(
        config={},
        region="us-chicago-1",
        compartment_id="ocid1.compartment.oc1..test",
        _client_factory=ErrorSpeechClient,
    )

    with pytest.raises(APIStatusError) as exc_info:
        async with provider.synthesize(
            "ola",
            conn_options=APIConnectOptions(max_retry=0, timeout=2),
        ) as stream:
            async for _ in stream:
                pass

    assert exc_info.value.status_code == 429
    assert exc_info.value.body == {"code": "TooManyRequests"}
    await provider.aclose()


class BlockingRawResponse(FakeRawResponse):
    def __init__(self, first_chunk: bytes) -> None:
        super().__init__([first_chunk])
        self.release = threading.Event()

    def stream(self, amount: int) -> Iterator[bytes]:
        self.stream_amounts.append(amount)
        yield self.chunks[0]
        self.release.wait(timeout=5)

    def close(self) -> None:
        self.closed = True
        self.release.set()


@pytest.mark.asyncio
async def test_oracle_tts_closes_response_when_cancelled() -> None:
    pcm = b"\x04\x00" * 24_000
    raw = BlockingRawResponse(pcm_as_wav(pcm))
    data = FakeStreamingData(raw)
    provider = oracle.TTS(
        config={},
        region="us-chicago-1",
        compartment_id="ocid1.compartment.oc1..test",
        samples_rate_in_hz=24_000,
        _client_factory=lambda: FakeSpeechClient(data),
    )

    stream = provider.synthesize(
        "ola",
        conn_options=APIConnectOptions(max_retry=0, timeout=2),
    )
    await asyncio.wait_for(anext(stream), timeout=2)
    await stream.aclose()

    assert raw.closed is True
    assert data.closed is True
    await provider.aclose()


@pytest.mark.parametrize("auth", ["instance_principal", "resource_principal"])
def test_oracle_tts_supports_principal_auth(
    monkeypatch: pytest.MonkeyPatch,
    auth: str,
) -> None:
    signer = SimpleNamespace(region="us-chicago-1")
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        oracle_tts.oci.auth.signers,
        "InstancePrincipalsSecurityTokenSigner",
        lambda: signer,
    )
    monkeypatch.setattr(
        oracle_tts.oci.auth.signers,
        "get_resource_principals_signer",
        lambda: signer,
    )

    def fake_client(config: dict[str, Any], **kwargs: Any) -> object:
        captured["config"] = config
        captured["kwargs"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(oracle_tts.ai_speech, "AIServiceSpeechClient", fake_client)

    oracle.TTS(
        config=None,
        auth=auth,  # type: ignore[arg-type]
        region="us-chicago-1",
        compartment_id="ocid1.compartment.oc1..test",
    )

    assert captured["config"] == {"region": "us-chicago-1"}
    assert captured["kwargs"]["signer"] is signer
    assert isinstance(captured["kwargs"]["retry_strategy"], oci.retry.NoneRetryStrategy)
