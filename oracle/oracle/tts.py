from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import oci
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from oci import ai_speech, retry

from .utils import validate_and_prepare_config

AuthMode = Literal["config", "instance_principal", "resource_principal"]


@dataclass
class _TTSOptions:
    compartment_id: str
    region: str
    model_name: Literal["TTS_2_NATURAL", "TTS_1_STANDARD"]
    model_family: Literal["ORACLE"]
    language_code: str
    voice_id: str
    speed: float
    samples_rate_in_hz: int
    output_format: Literal["PCM"]
    is_stream_enabled: bool
    chunk_size: int


class TTS(tts.TTS):
    """LiveKit OCI Speech TTS with chunked PCM output over HTTPS."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        compartment_id: str,
        region: str,
        auth: AuthMode = "config",
        config_file: str | None = None,
        profile: str = "DEFAULT",
        service_endpoint: str | None = None,
        model_name: Literal["TTS_2_NATURAL", "TTS_1_STANDARD"] = "TTS_2_NATURAL",
        model_family: Literal["ORACLE"] = "ORACLE",
        language_code: str = "pt-BR",
        voice_id: str = "Mariana",
        speed: float = 1.0,
        samples_rate_in_hz: int = 16_000,
        output_format: Literal["PCM"] = "PCM",
        is_stream_enabled: bool = True,
        chunk_size: int = 8_192,
        connect_timeout_s: float = 10.0,
        read_timeout_s: float = 60.0,
        _client_factory: Callable[[], Any] | None = None,
    ) -> None:
        if not compartment_id:
            raise ValueError("OCI Speech compartment_id is required")
        if model_name == "TTS_1_STANDARD":
            raise NotImplementedError(
                "This plugin does not support model type TTS_1_STANDARD yet"
            )
        if output_format != "PCM":
            raise ValueError("OCI Speech LiveKit streaming requires output_format='PCM'")
        if not is_stream_enabled:
            raise ValueError("OCI Speech chunk streaming requires is_stream_enabled=True")
        if not 8_000 <= samples_rate_in_hz <= 48_000:
            raise ValueError("samples_rate_in_hz must be between 8000 and 48000")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if connect_timeout_s <= 0 or read_timeout_s <= 0:
            raise ValueError("OCI Speech timeouts must be positive")

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=samples_rate_in_hz,
            num_channels=1,
        )
        self._client = (
            _client_factory()
            if _client_factory is not None
            else _build_client(
                config=config,
                region=region,
                auth=auth,
                config_file=config_file,
                profile=profile,
                service_endpoint=service_endpoint,
                connect_timeout_s=connect_timeout_s,
                read_timeout_s=read_timeout_s,
            )
        )
        self._opts = _TTSOptions(
            compartment_id=compartment_id,
            region=region,
            model_name=model_name,
            model_family=model_family,
            language_code=language_code,
            voice_id=voice_id,
            speed=speed,
            samples_rate_in_hz=samples_rate_in_hz,
            output_format=output_format,
            is_stream_enabled=True,
            chunk_size=chunk_size,
        )

    @property
    def provider(self) -> str:
        return "oci-speech"

    @property
    def model(self) -> str:
        return self._opts.model_name

    def update_options(
        self,
        *,
        language_code: NotGivenOr[str] = NOT_GIVEN,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(language_code):
            self._opts.language_code = language_code
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(speed):
            self._opts.speed = speed

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.SynthesizeStream:
        raise NotImplementedError(
            "OCI Speech streams audio chunks but does not accept incremental text; "
            "use synthesize() or LiveKit StreamAdapter"
        )

    async def aclose(self) -> None:
        session = getattr(getattr(self._client, "base_client", None), "session", None)
        close = getattr(session, "close", None)
        if callable(close):
            await asyncio.to_thread(close)


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._provider = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()
        stop = threading.Event()
        response_holder: dict[str, Any | None] = {"data": None}
        emitter_initialized = False

        def publish(item: bytes | BaseException | None) -> None:
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, item)

        def worker() -> None:
            pcm_decoder = _PcmWavHeaderStripper()
            try:
                response = self._provider._client.synthesize_speech(
                    ai_speech.models.SynthesizeSpeechDetails(
                        text=self._input_text,
                        is_stream_enabled=True,
                        compartment_id=self._opts.compartment_id,
                        configuration=ai_speech.models.TtsOracleConfiguration(
                            model_family=self._opts.model_family,
                            model_details=ai_speech.models.TtsOracleTts2NaturalModelDetails(
                                model_name=self._opts.model_name,
                                voice_id=self._opts.voice_id,
                                language_code=self._opts.language_code,
                            ),
                            speech_settings=ai_speech.models.TtsOracleSpeechSettings(
                                text_type="TEXT",
                                sample_rate_in_hz=self._opts.samples_rate_in_hz,
                                output_format="PCM",
                            ),
                        ),
                    )
                )
                response_data = response.data
                response_holder["data"] = response_data

                for chunk in _iter_response_bytes(response_data, self._opts.chunk_size):
                    if stop.is_set():
                        break
                    if not chunk:
                        continue
                    audio = pcm_decoder.feed(chunk)
                    if audio:
                        publish(audio)

                final_audio = pcm_decoder.finish()
                if final_audio:
                    publish(final_audio)
            except Exception as exc:
                publish(exc)
            finally:
                _close_response_data(response_holder["data"])
                publish(None)

        worker_task = asyncio.create_task(asyncio.to_thread(worker))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise _normalize_oci_exception(item) from item
                if not emitter_initialized:
                    output_emitter.initialize(
                        request_id=utils.shortuuid(),
                        sample_rate=self._opts.samples_rate_in_hz,
                        num_channels=1,
                        mime_type="audio/pcm",
                    )
                    emitter_initialized = True
                output_emitter.push(item)

            await worker_task
            if not emitter_initialized:
                raise APIConnectionError("OCI Speech returned no audio")
        finally:
            stop.set()
            _close_response_data(response_holder["data"])
            if not worker_task.done():
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(worker_task), timeout=0.5)


class _PcmWavHeaderStripper:
    """Remove one optional WAV envelope from an incremental PCM response."""

    _MAX_HEADER_BYTES = 1024 * 1024

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._passthrough = False

    def feed(self, chunk: bytes) -> bytes:
        if self._passthrough:
            return bytes(chunk)

        self._buffer.extend(chunk)
        if len(self._buffer) < 4:
            return b""
        if self._buffer[:4] != b"RIFF":
            return self._start_passthrough()

        if len(self._buffer) < 12:
            return b""
        if self._buffer[8:12] != b"WAVE":
            return self._start_passthrough()

        position = 12
        while True:
            if len(self._buffer) < position + 8:
                self._guard_header_size()
                return b""

            chunk_id = bytes(self._buffer[position : position + 4])
            chunk_size = int.from_bytes(
                self._buffer[position + 4 : position + 8],
                "little",
            )
            if chunk_id == b"data":
                audio_start = position + 8
                audio = bytes(self._buffer[audio_start:])
                self._buffer.clear()
                self._passthrough = True
                return audio

            next_position = position + 8 + chunk_size + (chunk_size & 1)
            if len(self._buffer) < next_position:
                self._guard_header_size()
                return b""
            position = next_position

    def finish(self) -> bytes:
        if self._passthrough or not self._buffer:
            return b""
        if len(self._buffer) < 4 or self._buffer[:4] != b"RIFF":
            return self._start_passthrough()
        raise APIConnectionError("OCI Speech returned an incomplete PCM WAV header")

    def _start_passthrough(self) -> bytes:
        audio = bytes(self._buffer)
        self._buffer.clear()
        self._passthrough = True
        return audio

    def _guard_header_size(self) -> None:
        if len(self._buffer) > self._MAX_HEADER_BYTES:
            raise APIConnectionError("OCI Speech PCM WAV header is too large")


def _build_client(
    *,
    config: dict[str, Any] | None,
    region: str,
    auth: AuthMode,
    config_file: str | None,
    profile: str,
    service_endpoint: str | None,
    connect_timeout_s: float,
    read_timeout_s: float,
) -> Any:
    oci_config, signer = _resolve_oci_auth(
        config=config,
        region=region,
        auth=auth,
        config_file=config_file,
        profile=profile,
    )
    kwargs: dict[str, Any] = {
        "retry_strategy": retry.NoneRetryStrategy(),
        "timeout": (connect_timeout_s, read_timeout_s),
    }
    if signer is not None:
        kwargs["signer"] = signer
    if service_endpoint:
        kwargs["service_endpoint"] = service_endpoint
    return ai_speech.AIServiceSpeechClient(oci_config, **kwargs)


def _resolve_oci_auth(
    *,
    config: dict[str, Any] | None,
    region: str,
    auth: AuthMode,
    config_file: str | None,
    profile: str,
) -> tuple[dict[str, Any], Any | None]:
    if auth == "instance_principal":
        if config is not None:
            raise ValueError("config cannot be combined with instance_principal auth")
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        return {"region": region or signer.region}, signer

    if auth == "resource_principal":
        if config is not None:
            raise ValueError("config cannot be combined with resource_principal auth")
        signer = oci.auth.signers.get_resource_principals_signer()
        resolved_region = region or getattr(signer, "region", None)
        if not resolved_region:
            raise ValueError("region is required for resource_principal auth")
        return {"region": resolved_region}, signer

    if auth != "config":
        raise ValueError(f"unsupported OCI auth mode: {auth}")

    if config is None:
        config = oci.config.from_file(
            file_location=config_file or oci.config.DEFAULT_LOCATION,
            profile_name=profile,
        )
    prepared = validate_and_prepare_config(config, region)
    return prepared, _security_token_signer_from_config(prepared)


def _security_token_signer_from_config(config: dict[str, Any]) -> Any | None:
    token_file = config.get("security_token_file")
    if not token_file:
        return None
    key_file = config.get("key_file")
    if not key_file:
        raise ValueError("OCI config has security_token_file but no key_file")
    private_key = oci.signer.load_private_key_from_file(
        key_file,
        pass_phrase=config.get("pass_phrase"),
    )
    token = Path(token_file).expanduser().read_text(encoding="utf-8").strip()
    return oci.auth.signers.SecurityTokenSigner(token, private_key)


def _iter_response_bytes(data: Any, chunk_size: int) -> Iterable[bytes]:
    if data is None:
        return
    if isinstance(data, (bytes, bytearray)):
        value = bytes(data)
        for offset in range(0, len(value), chunk_size):
            yield value[offset : offset + chunk_size]
        return

    raw = getattr(data, "raw", None)
    if raw is not None:
        if hasattr(raw, "stream"):
            yield from raw.stream(chunk_size)
            return
        if hasattr(raw, "read"):
            while chunk := raw.read(chunk_size):
                yield chunk
            return

    if hasattr(data, "stream"):
        yield from data.stream(chunk_size)
        return
    if hasattr(data, "iter_content"):
        yield from data.iter_content(chunk_size=chunk_size)
        return
    if hasattr(data, "read"):
        while chunk := data.read(chunk_size):
            yield chunk
        return
    raise TypeError(f"unsupported OCI Speech response type: {type(data).__name__}")


def _close_response_data(data: Any) -> None:
    if data is None:
        return

    seen: set[int] = set()
    for candidate in (getattr(data, "raw", None), data):
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        close = getattr(candidate, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()


def _normalize_oci_exception(exc: BaseException) -> BaseException:
    if isinstance(exc, oci.exceptions.ServiceError):
        return APIStatusError(
            exc.message or "OCI Speech request failed",
            status_code=exc.status,
            request_id=getattr(exc, "request_id", None),
            body={"code": exc.code},
        )
    if isinstance(exc, oci.exceptions.RequestException):
        return APIConnectionError(f"OCI Speech connection failed: {exc}")
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return APITimeoutError("OCI Speech request timed out")
    if isinstance(exc, (APIConnectionError, APIStatusError, APITimeoutError)):
        return exc
    return APIConnectionError(f"OCI Speech request failed: {type(exc).__name__}: {exc}")
