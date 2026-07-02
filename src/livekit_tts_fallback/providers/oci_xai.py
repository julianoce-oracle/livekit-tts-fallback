from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import websockets
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)

from ..capabilities import TransportCapabilities, TransportKind
from ..errors import ConfigurationError
from ..transports import AsyncConnectionPool, ConnectionPoolConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OciXaiConfig:
    region: str = "us-chicago-1"
    endpoint: str | None = None
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str = "OCI_XAI_API_KEY"
    voice: str = "ara"
    language: str = "auto"
    sample_rate: int = 24_000
    optimize_streaming_latency: int | None = None
    text_normalization: bool | None = None
    connect_timeout_s: float = 10.0
    first_audio_timeout_s: float = 10.0
    chunk_timeout_s: float = 10.0
    close_timeout_s: float = 2.0
    ping_interval_s: float = 20.0
    ping_timeout_s: float = 20.0
    extra_query: Mapping[str, str | int | bool] = field(default_factory=dict)
    pool: ConnectionPoolConfig = field(
        default_factory=lambda: ConnectionPoolConfig(min_size=1, max_size=3)
    )

    def __post_init__(self) -> None:
        if not self.region and not self.endpoint:
            raise ConfigurationError("OCI xAI region or endpoint is required")
        if self.sample_rate <= 0:
            raise ConfigurationError("OCI xAI sample_rate must be positive")
        if self.optimize_streaming_latency not in (None, 0, 1, 2):
            raise ConfigurationError(
                "OCI xAI optimize_streaming_latency must be 0, 1, 2, or None"
            )
        for name in (
            "connect_timeout_s",
            "first_audio_timeout_s",
            "chunk_timeout_s",
            "close_timeout_s",
            "ping_interval_s",
            "ping_timeout_s",
        ):
            if getattr(self, name) <= 0:
                raise ConfigurationError(f"OCI xAI {name} must be positive")

    @property
    def websocket_endpoint(self) -> str:
        return self.endpoint or (
            f"wss://inference.generativeai.{self.region}.oci.oraclecloud.com/xai/v1/tts"
        )

    def resolve_api_key(self) -> str:
        api_key = self.api_key or os.getenv(self.api_key_env)
        if not api_key:
            raise APIStatusError(
                f"OCI xAI API key is not configured; set {self.api_key_env}",
                status_code=401,
            )
        return api_key

    def websocket_url(self) -> str:
        query: dict[str, str | int] = {
            "voice": self.voice,
            "language": self.language,
            "codec": "pcm",
            "sample_rate": self.sample_rate,
        }
        if self.optimize_streaming_latency is not None:
            query["optimize_streaming_latency"] = self.optimize_streaming_latency
        if self.text_normalization is not None:
            query["text_normalization"] = str(self.text_normalization).lower()
        for key, value in self.extra_query.items():
            query[key] = str(value).lower() if isinstance(value, bool) else value
        return f"{self.websocket_endpoint.rstrip('/')}?{urlencode(query)}"


class _XaiWebSocketConnection:
    def __init__(self, websocket: Any, config: OciXaiConfig) -> None:
        self._websocket = websocket
        self._config = config

    @property
    def is_open(self) -> bool:
        closed = getattr(self._websocket, "closed", None)
        if isinstance(closed, bool):
            return not closed
        state_name = getattr(getattr(self._websocket, "state", None), "name", None)
        return state_name == "OPEN" if isinstance(state_name, str) else True

    async def close(self) -> None:
        await self._websocket.close()

    async def send_text(self, text: str) -> None:
        await self._send_json({"type": "text.delta", "delta": text})

    async def finish_text(self) -> None:
        await self._send_json({"type": "text.done"})

    async def receive_audio(self) -> AsyncIterator[bytes]:
        saw_audio = False
        while True:
            timeout_s = (
                self._config.chunk_timeout_s if saw_audio else self._config.first_audio_timeout_s
            )
            try:
                raw_message = await asyncio.wait_for(self._websocket.recv(), timeout=timeout_s)
            except TimeoutError as exc:
                phase = "next audio chunk" if saw_audio else "first audio chunk"
                raise APITimeoutError(f"OCI xAI timed out waiting for {phase}") from exc
            except websockets.exceptions.ConnectionClosed as exc:
                raise APIConnectionError(
                    f"OCI xAI websocket closed while receiving audio: code={exc.code}"
                ) from exc

            if not isinstance(raw_message, str):
                raise APIConnectionError("OCI xAI returned a non-JSON websocket message")

            try:
                event = json.loads(raw_message)
            except json.JSONDecodeError as exc:
                raise APIConnectionError("OCI xAI returned invalid JSON") from exc

            event_type = event.get("type")
            if event_type == "audio.delta":
                try:
                    chunk = base64.b64decode(event.get("delta", ""), validate=True)
                except (ValueError, TypeError) as exc:
                    raise APIConnectionError("OCI xAI returned invalid base64 audio") from exc
                if chunk:
                    saw_audio = True
                    yield chunk
                continue

            if event_type == "audio.done":
                return

            if event_type == "error":
                status_code = _coerce_status(event.get("status_code") or event.get("status"))
                raise APIStatusError(
                    str(event.get("message") or "OCI xAI synthesis failed"),
                    status_code=status_code,
                    request_id=event.get("request_id"),
                    body=event,
                )

            logger.debug("ignoring OCI xAI websocket event", extra={"event_type": event_type})

    async def _send_json(self, payload: Mapping[str, object]) -> None:
        try:
            await self._websocket.send(json.dumps(payload))
        except websockets.exceptions.ConnectionClosed as exc:
            raise APIConnectionError(
                f"OCI xAI websocket closed while sending text: code={exc.code}"
            ) from exc


class OciXaiTTS(tts.TTS):
    """OCI Generative AI xAI Voice provider using reusable WebSocket sessions."""

    transport_capabilities = TransportCapabilities(
        kind=TransportKind.WEBSOCKET,
        reusable_session=True,
        prewarm_supported=True,
        streaming_input=True,
    )

    def __init__(
        self,
        config: OciXaiConfig | None = None,
        *,
        _pool: AsyncConnectionPool[_XaiWebSocketConnection] | None = None,
    ) -> None:
        self.config = config or OciXaiConfig()
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=self.config.sample_rate,
            num_channels=1,
        )
        self._pool = _pool or AsyncConnectionPool(self._connect, config=self.config.pool)
        self._prewarm_task: asyncio.Task[None] | None = None

    @property
    def provider(self) -> str:
        return "oci-xai"

    @property
    def model(self) -> str:
        return "xai.grok-tts"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.SynthesizeStream:
        return _OciXaiSynthesizeStream(tts=self, conn_options=conn_options)

    def prewarm(self) -> None:
        if self._prewarm_task is not None and not self._prewarm_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("OCI xAI prewarm skipped because no event loop is running")
            return
        self._prewarm_task = loop.create_task(self._prewarm(), name="oci-xai-prewarm")

    async def _prewarm(self) -> None:
        try:
            await self._pool.start()
        except Exception:
            logger.warning("OCI xAI prewarm failed", exc_info=True)

    async def aclose(self) -> None:
        if self._prewarm_task is not None:
            if not self._prewarm_task.done():
                self._prewarm_task.cancel()
            await asyncio.gather(self._prewarm_task, return_exceptions=True)
        await self._pool.aclose()

    def pool_snapshot(self) -> dict[str, int | bool]:
        return self._pool.snapshot()

    async def _connect(self) -> _XaiWebSocketConnection:
        headers = {"Authorization": f"Bearer {self.config.resolve_api_key()}"}
        header_name = (
            "additional_headers"
            if "additional_headers" in inspect.signature(websockets.connect).parameters
            else "extra_headers"
        )
        started = time.perf_counter()
        try:
            websocket = await websockets.connect(
                self.config.websocket_url(),
                **{header_name: headers},
                open_timeout=self.config.connect_timeout_s,
                close_timeout=self.config.close_timeout_s,
                ping_interval=self.config.ping_interval_s,
                ping_timeout=self.config.ping_timeout_s,
                max_size=None,
            )
        except TimeoutError as exc:
            raise APITimeoutError("OCI xAI websocket connection timed out") from exc
        except Exception as exc:
            status_code = _status_code_from_exception(exc)
            if status_code is not None:
                raise APIStatusError(
                    "OCI xAI websocket handshake failed",
                    status_code=status_code,
                ) from exc
            raise APIConnectionError(
                f"OCI xAI websocket connection failed: {type(exc).__name__}: {exc}"
            ) from exc

        logger.debug(
            "OCI xAI websocket connected",
            extra={"connect_ms": round((time.perf_counter() - started) * 1000, 3)},
        )
        return _XaiWebSocketConnection(websocket, self.config)


class _OciXaiSynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: OciXaiTTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._provider = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        lease = None
        healthy = False
        acquire_started = time.perf_counter()
        try:
            lease = await self._provider._pool.acquire(
                timeout_s=min(
                    self._provider.config.pool.acquire_timeout_s,
                    self._conn_options.timeout,
                )
            )
            self._acquire_time = time.perf_counter() - acquire_started
            self._connection_reused = lease.reused
            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=self._provider.sample_rate,
                num_channels=self._provider.num_channels,
                mime_type="audio/pcm",
                stream=True,
            )

            segment_open = False
            async for item in self._input_ch:
                if isinstance(item, str):
                    if not segment_open:
                        output_emitter.start_segment(segment_id=utils.shortuuid())
                        self._mark_started()
                        segment_open = True
                    await lease.connection.send_text(item)
                    continue

                if not segment_open:
                    continue
                await lease.connection.finish_text()
                async for chunk in lease.connection.receive_audio():
                    output_emitter.push(chunk)
                output_emitter.end_segment()
                segment_open = False

            if segment_open:
                await lease.connection.finish_text()
                async for chunk in lease.connection.receive_audio():
                    output_emitter.push(chunk)
                output_emitter.end_segment()
            healthy = True
        except (APIConnectionError, APIStatusError, APITimeoutError):
            raise
        except TimeoutError as exc:
            raise APITimeoutError("OCI xAI connection pool timed out") from exc
        except Exception as exc:
            raise APIConnectionError(
                f"OCI xAI synthesis failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            if lease is not None:
                await self._provider._pool.release(lease, healthy=healthy)


def _status_code_from_exception(exc: BaseException) -> int | None:
    for candidate in (exc, getattr(exc, "response", None)):
        if candidate is None:
            continue
        for attribute in ("status_code", "status"):
            status = getattr(candidate, attribute, None)
            if isinstance(status, int):
                return status
    return None


def _coerce_status(value: object) -> int:
    try:
        return int(value) if value is not None else -1
    except (TypeError, ValueError):
        return -1
