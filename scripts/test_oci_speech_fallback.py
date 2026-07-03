from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any, cast

import oci
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    tts,
)

import oracle
from livekit_tts_fallback import (
    FallbackPolicy,
    OciXaiConfig,
    OciXaiTTS,
    build_fallback_tts,
)

logger = logging.getLogger("oci-speech-fallback-test")


class ForcedInternalFailureTTS(tts.TTS):
    """Primary used to reproduce an internal xAI failure without changing credentials."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24_000,
            num_channels=1,
        )
        self.calls = 0

    @property
    def provider(self) -> str:
        return "oci-xai-forced-internal-failure"

    @property
    def model(self) -> str:
        return "xai.grok-tts"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        self.calls += 1
        return ForcedFailureStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        return None


class ForcedFailureStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        raise APIConnectionError(
            "forced internal OCI xAI failure before first audio frame",
            retryable=False,
        )


class RecordingOracleTTS(oracle.TTS):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.synthesis_calls = 0

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        self.synthesis_calls += 1
        return super().synthesize(text, conn_options=conn_options)


def parse_args() -> argparse.Namespace:
    default_env = Path(__file__).resolve().parents[1] / ".env"

    parser = argparse.ArgumentParser(
        description="Validate LiveKit fallback from OCI xAI to real OCI Speech."
    )
    parser.add_argument("--env-file", type=Path, default=default_env)
    parser.add_argument(
        "--text",
        default="Olá. Este áudio confirma que o fallback para o OCI Speech funcionou.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/livekit-oci-speech-fallback.wav"),
    )
    parser.add_argument(
        "--real-primary",
        action="store_true",
        help="Call the real OCI xAI primary instead of forcing an internal failure.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"env file not found: {path}")

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip().rstrip("\r")
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable is empty: {name}")
    return value


def optional_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def livekit_output_format() -> str:
    configured = os.getenv("OCI_SPEECH_TTS_FORMAT", "PCM").strip().upper()
    if configured != "PCM":
        logger.warning(
            "overriding OCI_SPEECH_TTS_FORMAT=%s with PCM; "
            "oracle.TTS emits decoded PCM frames to LiveKit",
            configured,
        )
    return "PCM"


def build_oci_speech() -> RecordingOracleTTS:
    auth = required_env("OCI_SPEECH_AUTH").lower()
    if auth not in {"config", "instance_principal", "resource_principal"}:
        raise RuntimeError(f"unsupported OCI_SPEECH_AUTH: {auth}")

    config = None
    config_file = optional_env("OCI_SPEECH_CONFIG_FILE")
    profile = os.getenv("OCI_SPEECH_PROFILE", "DEFAULT")
    if auth == "config":
        config = oci.config.from_file(
            file_location=config_file or oci.config.DEFAULT_LOCATION,
            profile_name=profile,
        )

    return RecordingOracleTTS(
        config=config,
        auth=cast("Any", auth),
        config_file=config_file,
        profile=profile,
        region=os.getenv("OCI_SPEECH_REGION", "us-ashburn-1"),
        service_endpoint=optional_env("OCI_SPEECH_ENDPOINT"),
        compartment_id=required_env("OCI_SPEECH_COMPARTMENT_ID"),
        model_name=os.getenv("OCI_SPEECH_TTS_MODEL", "TTS_2_NATURAL"),
        voice_id=required_env("OCI_SPEECH_TTS_VOICE_ID"),
        language_code=os.getenv("OCI_SPEECH_TTS_LANGUAGE", "pt-BR"),
        samples_rate_in_hz=int(os.getenv("OCI_SPEECH_TTS_SAMPLE_RATE", "24000")),
        output_format=cast("Any", livekit_output_format()),
        is_stream_enabled=env_bool("OCI_SPEECH_TTS_STREAM_ENABLED", True),
        chunk_size=int(os.getenv("OCI_SPEECH_TTS_CHUNK_SIZE", "8192")),
        read_timeout_s=float(os.getenv("OCI_SPEECH_TTS_TIMEOUT", "30")),
        connect_timeout_s=float(os.getenv("OCI_SPEECH_TTS_CONNECT_TIMEOUT", "10")),
    )


def build_real_primary() -> OciXaiTTS:
    api_key = optional_env("OCI_XAI_API_KEY") or optional_env("OCI_GENAI_API_KEY")
    return OciXaiTTS(
        OciXaiConfig(
            api_key=api_key,
            api_key_env="OCI_XAI_API_KEY",
            region=os.getenv(
                "OCI_XAI_REGION",
                os.getenv("OCI_GENAI_REGION", "us-chicago-1"),
            ),
            endpoint=optional_env("OCI_XAI_ENDPOINT"),
            voice=os.getenv("OCI_XAI_VOICE", "ara"),
            language=os.getenv("OCI_XAI_LANGUAGE", "pt-BR"),
        )
    )


def save_wave(path: Path, frames: list[object], sample_rate: int, num_channels: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = b"".join(frame.data.tobytes() for frame in frames)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return len(pcm)


async def run(args: argparse.Namespace) -> int:
    load_env_file(args.env_file)
    fallback = build_oci_speech()
    primary: tts.TTS = build_real_primary() if args.real_primary else ForcedInternalFailureTTS()
    chain = build_fallback_tts(
        primary,
        fallbacks=[fallback],
        policy=FallbackPolicy(max_retry_per_tts=0, output_sample_rate=24_000),
    )

    def availability_changed(event: object) -> None:
        event_tts = getattr(event, "tts", None)
        logger.info(
            "provider_availability provider=%s available=%s",
            getattr(event_tts, "provider", "unknown"),
            getattr(event, "available", None),
        )

    chain.on("tts_availability_changed", availability_changed)
    logger.info(
        "test_started mode=%s env_file=%s fallback=oci-speech",
        "real-primary" if args.real_primary else "forced-internal-primary-failure",
        args.env_file,
    )
    logger.info("native LiveKit fallback is active; XAI_CIRCUIT_* variables are not used")

    started = time.perf_counter()
    frames = []
    try:
        async with chain.synthesize(
            args.text,
            conn_options=APIConnectOptions(
                max_retry=0,
                timeout=float(os.getenv("OCI_SPEECH_TTS_TIMEOUT", "30")) + 5,
            ),
        ) as stream:
            async for event in stream:
                frames.append(event.frame)

        if not frames:
            raise RuntimeError("LiveKit completed without producing audio frames")
        if fallback.synthesis_calls < 1 and not args.real_primary:
            raise RuntimeError("OCI Speech fallback was not invoked")

        winner = "oci-speech" if fallback.synthesis_calls else "oci-xai"
        byte_count = save_wave(
            args.output,
            frames,
            sample_rate=chain.sample_rate,
            num_channels=chain.num_channels,
        )
        logger.info(
            "RESULT=OK winner=%s elapsed_ms=%.3f frames=%d pcm_bytes=%d output=%s",
            winner,
            (time.perf_counter() - started) * 1000,
            len(frames),
            byte_count,
            args.output,
        )
        return 0
    finally:
        await chain.aclose()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return asyncio.run(run(args))
    except Exception:
        logger.exception("RESULT=ERROR OCI Speech fallback validation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
