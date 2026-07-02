from __future__ import annotations

import os

from livekit_tts_fallback import (
    FallbackPolicy,
    OciSpeechConfig,
    OciSpeechTTS,
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
    fallback = OciSpeechTTS(
        OciSpeechConfig(
            compartment_id=os.environ["OCI_SPEECH_COMPARTMENT_ID"],
            voice_id=os.environ["OCI_SPEECH_VOICE_ID"],
            region=os.getenv("OCI_SPEECH_REGION", "us-ashburn-1"),
            auth=os.getenv("OCI_SPEECH_AUTH", "config"),  # type: ignore[arg-type]
            config_file=os.getenv("OCI_SPEECH_CONFIG_FILE"),
            profile=os.getenv("OCI_SPEECH_PROFILE", "DEFAULT"),
        )
    )
    return build_fallback_tts(
        primary,
        fallbacks=[fallback],
        policy=FallbackPolicy(max_retry_per_tts=0),
    )
