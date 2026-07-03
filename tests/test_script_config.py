from __future__ import annotations

import logging

import pytest

from scripts.test_oci_speech_fallback import livekit_output_format


def test_livekit_output_format_keeps_pcm(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("OCI_SPEECH_TTS_FORMAT", "PCM")

    assert livekit_output_format() == "PCM"
    assert not caplog.records


def test_livekit_output_format_overrides_legacy_mp3(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("OCI_SPEECH_TTS_FORMAT", "mp3")

    with caplog.at_level(logging.WARNING, logger="oci-speech-fallback-test"):
        assert livekit_output_format() == "PCM"

    assert "overriding OCI_SPEECH_TTS_FORMAT=MP3 with PCM" in caplog.text
