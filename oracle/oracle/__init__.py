from __future__ import annotations

from typing import Any

from .tts import TTS

__all__ = ["STT", "TTS", "LLM", "LLMStream"]


def __getattr__(name: str) -> Any:
    if name == "STT":
        from .stt import STT

        return STT
    if name in {"LLM", "LLMStream"}:
        from .llm import LLM, LLMStream

        return {"LLM": LLM, "LLMStream": LLMStream}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
