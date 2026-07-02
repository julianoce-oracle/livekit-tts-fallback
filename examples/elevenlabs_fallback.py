from __future__ import annotations

import os

from livekit_tts_fallback import OciXaiTTS, build_fallback_tts
from livekit_tts_fallback.providers import create_elevenlabs_tts


def build_tts():
    primary = OciXaiTTS()
    fallback = create_elevenlabs_tts(
        api_key=os.environ["ELEVEN_API_KEY"],
        voice_id=os.environ["ELEVENLABS_VOICE_ID"],
        model="eleven_multilingual_v2",
    )
    return build_fallback_tts(primary, fallbacks=[fallback])
