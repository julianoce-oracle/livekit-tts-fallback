from __future__ import annotations


class TTSFallbackError(Exception):
    """Base error raised by this package outside provider requests."""


class ConfigurationError(TTSFallbackError, ValueError):
    """The provider or fallback chain configuration is invalid."""


class OptionalDependencyError(TTSFallbackError, ImportError):
    """An optional provider dependency is not installed."""
