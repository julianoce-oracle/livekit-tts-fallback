from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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

logger = logging.getLogger(__name__)

OciAuthMode = Literal["config", "instance_principal", "resource_principal"]


@dataclass(frozen=True, slots=True)
class OciSpeechConfig:
    compartment_id: str
    voice_id: str
    region: str | None = None
    service_endpoint: str | None = None
    auth: OciAuthMode = "config"
    config_file: str | None = None
    profile: str = "DEFAULT"
    language_code: str = "pt-BR"
    model_name: str = "TTS_2_NATURAL"
    output_format: str = "wav"
    sample_rate: int = 24_000
    stream_enabled: bool = True
    chunk_size: int = 16_384
    max_concurrency: int = 4
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not self.compartment_id:
            raise ConfigurationError("OCI Speech compartment_id is required")
        if not self.voice_id:
            raise ConfigurationError("OCI Speech voice_id is required")
        if self.auth not in {"config", "instance_principal", "resource_principal"}:
            raise ConfigurationError(f"unsupported OCI Speech auth mode: {self.auth}")
        if self.sample_rate <= 0:
            raise ConfigurationError("OCI Speech sample_rate must be positive")
        if self.chunk_size < 1:
            raise ConfigurationError("OCI Speech chunk_size must be positive")
        if self.max_concurrency < 1:
            raise ConfigurationError("OCI Speech max_concurrency must be positive")
        if self.output_format.lower().lstrip(".") not in {"wav", "mp3"}:
            raise ConfigurationError("OCI Speech output_format must be wav or mp3")

    @property
    def normalized_output_format(self) -> str:
        return self.output_format.lower().lstrip(".")


class OciSpeechTTS(tts.TTS):
    """OCI Speech provider using one HTTPS synthesis request per utterance."""

    transport_capabilities = TransportCapabilities(
        kind=TransportKind.HTTPS,
        reusable_session=False,
        prewarm_supported=True,
        streaming_input=False,
    )

    def __init__(
        self,
        config: OciSpeechConfig,
        *,
        _client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=config.sample_rate,
            num_channels=1,
        )
        self._client_factory = _client_factory or self._build_client
        self._client: Any | None = None
        self._client_lock = threading.Lock()
        self._concurrency = asyncio.Semaphore(config.max_concurrency)
        self._prewarm_task: asyncio.Task[Any] | None = None

    @property
    def provider(self) -> str:
        return "oci-speech"

    @property
    def model(self) -> str:
        return self.config.model_name

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _OciSpeechChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    def prewarm(self) -> None:
        if self._prewarm_task is not None and not self._prewarm_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("OCI Speech prewarm skipped because no event loop is running")
            return
        self._prewarm_task = loop.create_task(
            asyncio.to_thread(self._get_client),
            name="oci-speech-prewarm",
        )

    async def aclose(self) -> None:
        if self._prewarm_task is not None:
            await asyncio.gather(self._prewarm_task, return_exceptions=True)
        client = self._client
        self._client = None
        if client is None:
            return

        session = getattr(getattr(client, "base_client", None), "session", None)
        close = getattr(session, "close", None)
        if callable(close):
            await asyncio.to_thread(close)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                self._client = self._client_factory()
        return self._client

    def _build_client(self) -> Any:
        import oci
        from oci.ai_speech import AIServiceSpeechClient

        signer = None
        if self.config.auth == "instance_principal":
            signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
            oci_config: dict[str, Any] = {"region": self.config.region or signer.region}
        elif self.config.auth == "resource_principal":
            signer = oci.auth.signers.get_resource_principals_signer()
            region = self.config.region or getattr(signer, "region", None)
            if not region:
                raise ConfigurationError(
                    "OCI Speech region is required when resource principal has no region"
                )
            oci_config = {"region": region}
        else:
            config_file = self.config.config_file or oci.config.DEFAULT_LOCATION
            oci_config = oci.config.from_file(
                file_location=config_file,
                profile_name=self.config.profile,
            )
            if self.config.region:
                oci_config["region"] = self.config.region
            signer = _security_token_signer_from_config(oci, oci_config)

        client_kwargs: dict[str, Any] = {
            "timeout": (self.config.connect_timeout_s, self.config.read_timeout_s)
        }
        if signer is not None:
            client_kwargs["signer"] = signer
        if self.config.service_endpoint:
            client_kwargs["service_endpoint"] = self.config.service_endpoint
        return AIServiceSpeechClient(oci_config, **client_kwargs)

    def _synthesize_response(self, text: str) -> Any:
        from oci.ai_speech import models

        details = models.SynthesizeSpeechDetails(
            text=text,
            is_stream_enabled=self.config.stream_enabled,
            compartment_id=self.config.compartment_id,
            configuration=models.TtsOracleConfiguration(
                model_details=_build_model_details(models, self.config)
            ),
            audio_config=models.TtsBaseAudioConfig(
                save_path=f"response.{self.config.normalized_output_format}"
            ),
        )
        return self._get_client().synthesize_speech(details)


class _OciSpeechChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: OciSpeechTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._provider = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()
        stop = threading.Event()
        emitter_initialized = False

        def publish(item: bytes | BaseException | None) -> None:
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, item)

        def worker() -> None:
            try:
                response = self._provider._synthesize_response(self._input_text)
                for chunk in _iter_response_bytes(
                    response.data,
                    self._provider.config.chunk_size,
                ):
                    if stop.is_set():
                        break
                    if chunk:
                        publish(chunk)
            except BaseException as exc:
                publish(exc)
            finally:
                publish(None)

        async with self._provider._concurrency:
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
                            sample_rate=self._provider.sample_rate,
                            num_channels=self._provider.num_channels,
                            mime_type=_detect_audio_mime_type(
                                item,
                                self._provider.config.normalized_output_format,
                            ),
                            stream=False,
                        )
                        emitter_initialized = True
                    output_emitter.push(item)
                await worker_task
                if emitter_initialized:
                    output_emitter.flush()
            finally:
                stop.set()
                if not worker_task.done():
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(asyncio.shield(worker_task), timeout=0.5)


def _detect_audio_mime_type(first_chunk: bytes, configured_format: str) -> str:
    if first_chunk[:4] == bytes((82, 73, 70, 70)) and first_chunk[8:12] == bytes((87, 65, 86, 69)):
        return "audio/wav"
    if first_chunk[:3] == bytes((73, 68, 51)) or (
        len(first_chunk) >= 2 and first_chunk[0] == 0xFF and first_chunk[1] & 0xE0 == 0xE0
    ):
        return "audio/mpeg"
    return "audio/mpeg" if configured_format == "mp3" else "audio/pcm"


def _security_token_signer_from_config(oci_module: Any, config: dict[str, Any]) -> Any | None:
    token_file = config.get("security_token_file")
    if not token_file:
        return None
    key_file = config.get("key_file")
    if not key_file:
        raise ConfigurationError("OCI config has security_token_file but no key_file")
    private_key = oci_module.signer.load_private_key_from_file(
        key_file,
        pass_phrase=config.get("pass_phrase"),
    )
    token = Path(token_file).read_text(encoding="utf-8").strip()
    return oci_module.auth.signers.SecurityTokenSigner(token, private_key)


def _build_model_details(models: Any, config: OciSpeechConfig) -> Any:
    if config.model_name.upper() == "TTS_1_STANDARD":
        return models.TtsOracleTts1StandardModelDetails(voice_id=config.voice_id)

    kwargs: dict[str, str] = {"voice_id": config.voice_id}
    supported_fields = getattr(models.TtsOracleTts2NaturalModelDetails(), "swagger_types", {})
    if "language_code" in supported_fields:
        kwargs["language_code"] = config.language_code
    return models.TtsOracleTts2NaturalModelDetails(**kwargs)


def _iter_response_bytes(data: Any, chunk_size: int) -> Iterable[bytes]:
    if data is None:
        return
    if isinstance(data, bytes):
        yield data
        return
    if isinstance(data, bytearray):
        yield bytes(data)
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
    if isinstance(data, str):
        with Path(data).open("rb") as audio_file:
            while chunk := audio_file.read(chunk_size):
                yield chunk
        return
    raise TypeError(f"unsupported OCI Speech response type: {type(data).__name__}")


def _normalize_oci_exception(exc: BaseException) -> BaseException:
    try:
        import oci

        if isinstance(exc, oci.exceptions.ServiceError):
            return APIStatusError(
                exc.message or "OCI Speech request failed",
                status_code=exc.status,
                request_id=exc.request_id,
                body={"code": exc.code},
            )
    except ImportError:
        pass

    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return APITimeoutError("OCI Speech request timed out")
    if isinstance(exc, (APIConnectionError, APIStatusError, APITimeoutError)):
        return exc
    return APIConnectionError(f"OCI Speech request failed: {type(exc).__name__}: {exc}")
