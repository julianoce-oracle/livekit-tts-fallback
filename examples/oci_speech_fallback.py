from __future__ import annotations

import os

import oci

import oracle
from livekit_tts_fallback import (
    FallbackPolicy,
    OciXaiConfig,
    OciXaiTTS,
    build_fallback_tts,
)


def build_tts():
    primary = OciXaiTTS(
        OciXaiConfig(
            region=os.getenv("OCI_XAI_REGION", "us-chicago-1"),
            voice=os.getenv("OCI_XAI_VOICE", "ara"),
            language=os.getenv("OCI_XAI_LANGUAGE", "pt-BR"),
        )
    )

    profile = os.getenv("OCI_SPEECH_PROFILE", "DEFAULT")
    config_file = os.getenv("OCI_SPEECH_CONFIG_FILE") or oci.config.DEFAULT_LOCATION
    config = oci.config.from_file(
        file_location=config_file,
        profile_name=profile,
    )
    fallback = oracle.TTS(
        config=config,
        compartment_id=os.environ["OCI_SPEECH_COMPARTMENT_ID"],
        region=os.getenv("OCI_SPEECH_REGION", "us-ashburn-1"),
        voice_id=os.environ["OCI_SPEECH_TTS_VOICE_ID"],
        language_code=os.getenv("OCI_SPEECH_TTS_LANGUAGE", "pt-BR"),
        samples_rate_in_hz=int(os.getenv("OCI_SPEECH_TTS_SAMPLE_RATE", "24000")),
    )
    return build_fallback_tts(
        primary,
        fallbacks=[fallback],
        policy=FallbackPolicy(max_retry_per_tts=0),
    )
