from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import pytest

from livekit_tts_fallback import OciSpeechConfig, OciSpeechTTS


class FakeSpeechClient:
    def __init__(self, audio: bytes) -> None:
        self.audio = audio
        self.calls = []

    def synthesize_speech(self, details: object) -> object:
        self.calls.append(details)
        return SimpleNamespace(data=self.audio)


def wav_silence() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\x00\x00" * 4_800)
    return output.getvalue()


@pytest.mark.asyncio
async def test_oci_speech_decodes_https_audio_for_livekit() -> None:
    client = FakeSpeechClient(wav_silence())
    provider = OciSpeechTTS(
        OciSpeechConfig(
            compartment_id="ocid1.compartment.oc1..test",
            voice_id="test-voice",
        ),
        _client_factory=lambda: client,
    )

    async with provider.synthesize("ola") as stream:
        events = [event async for event in stream]

    assert events
    assert len(client.calls) == 1
    await provider.aclose()
