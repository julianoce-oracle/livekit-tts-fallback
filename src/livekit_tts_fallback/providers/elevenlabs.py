from __future__ import annotations

from typing import Any

from livekit.agents import tts

from ..errors import OptionalDependencyError


def create_elevenlabs_tts(**kwargs: Any) -> tts.TTS:
    """Create the official LiveKit ElevenLabs plugin only when selected by the user."""

    try:
        from livekit.plugins import elevenlabs
    except ImportError as exc:
        raise OptionalDependencyError(
            "ElevenLabs support is optional; install livekit-tts-fallback[elevenlabs]"
        ) from exc
    return elevenlabs.TTS(**kwargs)
