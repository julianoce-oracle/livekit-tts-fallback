from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TransportKind(StrEnum):
    WEBSOCKET = "websocket"
    HTTPS = "https"
    PROVIDER_MANAGED = "provider_managed"


@dataclass(frozen=True, slots=True)
class TransportCapabilities:
    """Informational transport metadata; fallback decisions remain LiveKit's concern."""

    kind: TransportKind
    reusable_session: bool
    prewarm_supported: bool
    streaming_input: bool
