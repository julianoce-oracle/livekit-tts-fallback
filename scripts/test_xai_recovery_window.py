from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    tts,
)
from test_oci_speech_fallback import (
    RecordingOracleTTS,
    build_oci_speech,
    build_real_primary,
    load_env_file,
    save_wave,
)

from livekit_tts_fallback import FallbackPolicy, OciXaiTTS, build_fallback_tts

logger = logging.getLogger("xai-recovery-window-test")


class WindowedFailureTTS(tts.TTS):
    """Blocks the real primary for a fixed interval without changing credentials."""

    def __init__(self, delegate: OciXaiTTS, *, recover_after_s: float) -> None:
        super().__init__(
            capabilities=delegate.capabilities,
            sample_rate=delegate.sample_rate,
            num_channels=delegate.num_channels,
        )
        self._delegate = delegate
        self._started_at = time.perf_counter()
        self._recover_at = self._started_at + recover_after_s
        self.calls = 0
        self.forced_failures = 0
        self.real_calls = 0

    @property
    def provider(self) -> str:
        return self._delegate.provider

    @property
    def model(self) -> str:
        return self._delegate.model

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._started_at

    @property
    def gate_remaining_s(self) -> float:
        return max(0.0, self._recover_at - time.perf_counter())

    @property
    def is_gate_open(self) -> bool:
        return self.gate_remaining_s <= 0.0

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        self.calls += 1
        if not self.is_gate_open:
            self.forced_failures += 1
            return WindowedFailureStream(
                tts=self,
                input_text=text,
                conn_options=conn_options,
            )

        self.real_calls += 1
        return self._delegate.synthesize(text, conn_options=conn_options)

    def prewarm(self) -> None:
        if self.is_gate_open:
            self._delegate.prewarm()

    async def aclose(self) -> None:
        await self._delegate.aclose()


class WindowedFailureStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        provider = self._tts
        assert isinstance(provider, WindowedFailureTTS)
        raise APIConnectionError(
            "forced internal OCI xAI outage window is active "
            f"(remaining={provider.gate_remaining_s:.3f}s)",
            retryable=False,
        )


@dataclass(frozen=True, slots=True)
class RequestResult:
    winner: str
    frames: list[object]
    elapsed_ms: float


def parse_args() -> argparse.Namespace:
    default_env = Path(__file__).resolve().parents[1] / ".env"

    parser = argparse.ArgumentParser(
        description=(
            "Force a temporary internal OCI xAI outage, probe recovery through "
            "LiveKit, and verify routing returns to the real xAI WebSocket."
        )
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=default_env,
        help="Environment file containing OCI xAI and OCI Speech settings.",
    )
    parser.add_argument(
        "--monitor-seconds",
        type=float,
        default=30.0,
        help="Maximum recovery monitoring window.",
    )
    parser.add_argument(
        "--recover-after-seconds",
        type=float,
        default=12.0,
        help="Time until the internal failure gate permits real xAI calls.",
    )
    parser.add_argument(
        "--probe-interval-seconds",
        type=float,
        default=3.0,
        help="Delay between LiveKit synthesis requests that trigger recovery probes.",
    )
    parser.add_argument(
        "--text",
        default="Olá. Esta fala confirma o provider selecionado.",
    )
    parser.add_argument(
        "--probe-text",
        default="Teste curto de recuperacao do xAI.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/livekit-xai-recovery-window"),
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--verbose-livekit",
        action="store_true",
        help="Show expected LiveKit fallback and recovery tracebacks.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.monitor_seconds <= 0:
        parser.error("--monitor-seconds must be positive")
    if args.recover_after_seconds <= 0:
        parser.error("--recover-after-seconds must be positive")
    if args.recover_after_seconds >= args.monitor_seconds:
        parser.error("--recover-after-seconds must be smaller than --monitor-seconds")
    if args.probe_interval_seconds <= 0:
        parser.error("--probe-interval-seconds must be positive")
    if args.request_timeout_seconds is not None and args.request_timeout_seconds <= 0:
        parser.error("--request-timeout-seconds must be positive")

    return args


async def synthesize_once(
    chain: tts.TTS,
    *,
    primary: WindowedFailureTTS,
    fallback: RecordingOracleTTS,
    text: str,
    timeout_s: float,
) -> RequestResult:
    fallback_calls_before = fallback.synthesis_calls
    real_calls_before = primary.real_calls
    started = time.perf_counter()
    frames: list[object] = []

    async with chain.synthesize(
        text,
        conn_options=APIConnectOptions(max_retry=0, timeout=timeout_s),
    ) as stream:
        async for event in stream:
            frames.append(event.frame)

    if not frames:
        raise RuntimeError("LiveKit completed without producing audio frames")

    if fallback.synthesis_calls > fallback_calls_before:
        winner = "oci-speech"
    elif primary.real_calls > real_calls_before:
        winner = "oci-xai"
    else:
        winner = "unknown"

    return RequestResult(
        winner=winner,
        frames=frames,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


async def wait_for_recovery_or_timeout(
    recovered: asyncio.Event,
    *,
    timeout_s: float,
) -> bool:
    try:
        await asyncio.wait_for(recovered.wait(), timeout=timeout_s)
    except TimeoutError:
        return False
    return True


async def run(args: argparse.Namespace) -> int:
    load_env_file(args.env_file)
    request_timeout_s = args.request_timeout_seconds or (
        float(os.getenv("OCI_SPEECH_TTS_TIMEOUT", "30")) + 5.0
    )

    real_primary = build_real_primary()
    primary = WindowedFailureTTS(
        real_primary,
        recover_after_s=args.recover_after_seconds,
    )
    fallback = build_oci_speech()
    chain = build_fallback_tts(
        primary,
        fallbacks=[fallback],
        policy=FallbackPolicy(max_retry_per_tts=0, output_sample_rate=24_000),
    )
    recovered = asyncio.Event()

    def availability_changed(event: object) -> None:
        event_tts = getattr(event, "tts", None)
        available = getattr(event, "available", None)
        logger.info(
            "provider_availability t=%.3fs provider=%s available=%s gate_remaining=%.3fs",
            primary.elapsed_s,
            getattr(event_tts, "provider", "unknown"),
            available,
            primary.gate_remaining_s,
        )
        if event_tts is primary and available is True:
            recovered.set()

    chain.on("tts_availability_changed", availability_changed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_output = args.output_dir / "01-initial-oci-speech.wav"
    restored_output = args.output_dir / "02-restored-oci-xai.wav"

    logger.info(
        "test_started monitor_seconds=%.3f recover_after_seconds=%.3f "
        "probe_interval_seconds=%.3f env_file=%s",
        args.monitor_seconds,
        args.recover_after_seconds,
        args.probe_interval_seconds,
        args.env_file,
    )
    logger.info(
        "failure_mode=internal-timed-gate credentials_unchanged=true "
        "recovery_probe=livekit-synthesis-request"
    )

    probe_count = 0
    monitor_started = 0.0
    try:
        initial = await synthesize_once(
            chain,
            primary=primary,
            fallback=fallback,
            text=args.text,
            timeout_s=request_timeout_s,
        )
        if initial.winner != "oci-speech":
            raise RuntimeError(f"expected initial OCI Speech fallback, got winner={initial.winner}")

        initial_bytes = save_wave(
            initial_output,
            initial.frames,
            sample_rate=chain.sample_rate,
            num_channels=chain.num_channels,
        )
        logger.info(
            "initial_request winner=%s elapsed_ms=%.3f frames=%d pcm_bytes=%d output=%s",
            initial.winner,
            initial.elapsed_ms,
            len(initial.frames),
            initial_bytes,
            initial_output,
        )

        monitor_started = time.perf_counter()
        deadline = monitor_started + args.monitor_seconds
        while not recovered.is_set():
            remaining_s = deadline - time.perf_counter()
            if remaining_s <= 0:
                break

            wait_s = min(args.probe_interval_seconds, remaining_s)
            if await wait_for_recovery_or_timeout(recovered, timeout_s=wait_s):
                break

            probe_count += 1
            probe = await synthesize_once(
                chain,
                primary=primary,
                fallback=fallback,
                text=args.probe_text,
                timeout_s=request_timeout_s,
            )
            logger.info(
                "probe_result attempt=%d t=%.3fs request_winner=%s elapsed_ms=%.3f "
                "gate_open=%s gate_remaining=%.3fs real_xai_calls=%d oci_calls=%d",
                probe_count,
                primary.elapsed_s,
                probe.winner,
                probe.elapsed_ms,
                primary.is_gate_open,
                primary.gate_remaining_s,
                primary.real_calls,
                fallback.synthesis_calls,
            )

        if not recovered.is_set():
            raise TimeoutError(
                "OCI xAI did not recover within the monitoring window "
                f"({args.monitor_seconds:.3f}s)"
            )

        final_result = await synthesize_once(
            chain,
            primary=primary,
            fallback=fallback,
            text=args.text,
            timeout_s=request_timeout_s,
        )
        if final_result.winner != "oci-xai":
            raise RuntimeError(
                f"expected routing to return to OCI xAI, got winner={final_result.winner}"
            )

        final_bytes = save_wave(
            restored_output,
            final_result.frames,
            sample_rate=chain.sample_rate,
            num_channels=chain.num_channels,
        )
        monitor_elapsed_s = time.perf_counter() - monitor_started
        logger.info(
            "final_request winner=%s elapsed_ms=%.3f frames=%d pcm_bytes=%d output=%s",
            final_result.winner,
            final_result.elapsed_ms,
            len(final_result.frames),
            final_bytes,
            restored_output,
        )
        logger.info(
            "RESULT=OK monitor_elapsed_s=%.3f probes=%d forced_failures=%d "
            "real_xai_calls=%d oci_calls=%d route=oci-speech -> oci-xai",
            monitor_elapsed_s,
            probe_count,
            primary.forced_failures,
            primary.real_calls,
            fallback.synthesis_calls,
        )
        return 0
    finally:
        chain.off("tts_availability_changed", availability_changed)
        await chain.aclose()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.verbose_livekit:
        logging.getLogger("livekit.agents").setLevel(logging.ERROR)
    try:
        return asyncio.run(run(args))
    except Exception:
        logger.exception("RESULT=ERROR xAI recovery window validation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
