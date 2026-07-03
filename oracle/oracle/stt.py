import asyncio
import json
import time
import weakref
from typing import Any, Literal
from urllib.parse import urlencode

import aiohttp
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer
from oci import ai_speech, retry

from .utils import logger, validate_and_prepare_config

_max_session_duration = 10 * 60
# emit interim transcriptions every 0.5 seconds
_delta_transcript_interval = 0.5
SAMPLE_RATE = 16000
NUM_CHANNELS = 1


class STT(stt.STT):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        compartment_id: str,
        region: str,
        model_type: Literal["ORACLE", "WHISPER"] = "WHISPER",
        language: str = "pt",
        punctuation: Literal["AUTO", "NONE"] = "NONE",
    ):
        if model_type == "ORACLE":
            raise NotImplementedError(
                "This plugin does not support model type Oracle yet"
            )
        config = validate_and_prepare_config(config, region)
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=True)
        )
        self._client = ai_speech.AIServiceSpeechClient(
            config=config,
            # TODO: review a more adequate retry strategy and timeout
            retry_strategy=retry.NoneRetryStrategy(),
            timeout=(10, 60),
        )
        self.compartment_id = compartment_id
        self.model_type = model_type
        self.language = language
        self.region = region
        self.punctuation = punctuation

        self._streams = weakref.WeakSet[SpeechStream]()
        self._session: aiohttp.ClientSession | None = None
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            max_session_duration=_max_session_duration,
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
        )
        self._start_time = 0

    def create_token(self) -> str:
        session_token_details = ai_speech.models.CreateRealtimeSessionTokenDetails()
        session_token_details.compartment_id = self.compartment_id
        token_response = self._client.create_realtime_session_token(
            session_token_details
        )
        print(token_response)
        if not token_response:
            raise Exception("Empty token response")
        if token_response.status != 200:
            raise Exception("Error getting token")

        return token_response.data.token

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        token = self.create_token()
        realtime_config: dict[str, Any] = {
            "authenticationType": "TOKEN",
            "token": token,
            "compartmentId": self.compartment_id,
        }

        query_params: dict[str, str] = {
            "punctuation": self.punctuation,
            "languageCode": self.language,
            "modelDomain": "GENERIC",
            "modelType": self.model_type,
        }
        url = f"{str(self._base_url).rstrip('/')}?{urlencode(query_params)}"
        session = self._ensure_session()
        headers = {
            "User-Agent": "LiveKit Agents",
        }
        ws = await asyncio.wait_for(session.ws_connect(url, headers=headers), timeout)
        await ws.send_json(realtime_config)
        connect_response = await ws.receive(timeout=5)
        self._start_time = time.time()
        data = json.loads(connect_response.data)
        if data.get("event", None) != "CONNECT":
            raise Exception("Unable to connect to Oracle Live Transcribe")
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    @property
    def _base_url(self):
        return f"wss://realtime.aiservice.{self.region}.oci.oraclecloud.com/ws/transcribe/stream"

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechStream:
        # TODO: converter language code do livekit para OCI
        if language:
            self.language = language
        stream = SpeechStream(
            stt=self,
            pool=self._pool,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ):
        raise NotImplementedError("Only streaming is supported by this STT")


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        pool: utils.ConnectionPool[aiohttp.ClientWebSocketResponse],
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._stt = stt
        self._pool = pool
        self._language = stt.language
        self._request_id = ""
        self._reconnect_event = asyncio.Event()

    def update_options(
        self,
        *,
        language: str,
    ) -> None:
        self._language = language
        self._pool.invalidate()
        self._reconnect_event.set()

    @utils.log_exceptions(logger=logger)
    async def _run(self) -> None:
        closing_ws = False

        # async def keepalive_task(ws: aiohttp.ClientWebSocketResponse) -> None:
        #     # if we want to keep the connection alive even if no audio is sent,
        #     # Deepgram expects a keepalive message.
        #     # https://developers.deepgram.com/reference/listen-live#stream-keepalive
        #     try:
        #         while True:
        #             await ws.send_str(SpeechStream._KEEPALIVE_MSG)
        #             await asyncio.sleep(5)
        #     except Exception:
        #         return

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=SAMPLE_RATE // 20,
            )

            async for data in self._input_ch:
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    frames.extend(audio_bstream.write(data.data.tobytes()))
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())

                for frame in frames:
                    # # Checksum
                    # print(sum(frame.data.tobytes()) & 0xFFFF)
                    await ws.send_bytes(frame.data.tobytes())

            closing_ws = True

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            current_text = ""
            last_interim_at: float = 0
            connected_at = time.time()
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:  # close is expected, see SpeechStream.aclose
                        return

                    # this will trigger a reconnection, see the _run loop
                    raise APIStatusError(
                        "Oracle Live Transcribe connection closed unexpectedly"
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("Unexpected Oracle message type %s", msg.type)
                    continue

                try:
                    data = json.loads(msg.data)
                    event = data.get("event")

                    if event == "RESULT":
                        # TODO: review if it's safe to always use index 0
                        transcription = data.get("transcriptions")[0]
                        is_final = transcription.get("isFinal")
                        if not is_final:
                            delta = transcription.get("transcription", "")
                            if delta:
                                current_text += delta
                                if (
                                    time.time() - last_interim_at
                                    > _delta_transcript_interval
                                ):
                                    self._event_ch.send_nowait(
                                        stt.SpeechEvent(
                                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                            alternatives=[
                                                stt.SpeechData(
                                                    text=current_text,
                                                    language=self._language,
                                                )
                                            ],
                                        )
                                    )
                                    last_interim_at = time.time()
                        else:
                            current_text = ""
                            transcript = transcription.get("transcription", "")
                            print(
                                f"Transcription confidence: {transcription.get('confidence', None)}",
                            )
                            print(
                                f"Transcription: {transcript}",
                            )
                            if self._language == "pt":
                                if (
                                    str(transcript).lower() == "e aí"
                                    and float(transcription.get("confidence", 100.0))
                                    <= 50.0
                                ):
                                    # TODO: incrementar algum contador pra mostrar possível nível de ruído
                                    continue

                            if transcript:
                                duration = (
                                    float(transcription.get("endTimeInMs", 0))
                                    - float(transcription.get("startTimeInMs", 0))
                                ) / 1000
                                self._event_ch.send_nowait(
                                    stt.SpeechEvent(
                                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                        alternatives=[
                                            stt.SpeechData(
                                                text=transcript,
                                                language=self._language,
                                                start_time=self._stt._start_time
                                                - duration,
                                                end_time=self._stt._start_time
                                                + duration,
                                                confidence=float(
                                                    transcription.get("confidence", 100)
                                                )
                                                / 100,
                                            )
                                        ],
                                    )
                                )
                            # restart session if needed
                            if time.time() - connected_at > _max_session_duration:
                                logger.info(
                                    "Resetting Live Transcribe session due to timeout"
                                )
                                self._pool.remove(ws)
                                self._reconnect_event.set()
                                return
                    else:
                        logger.warning("Unexpected Oracle event type %s", event)
                        continue
                except Exception:
                    logger.exception("Failed to process Oracle message")

        while True:
            closing_ws = False  # reset the flag
            async with self._pool.connection(timeout=self._conn_options.timeout) as ws:
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # propagate exceptions from completed tasks
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    await self._pool.aclose()
                    tasks_group.exception()  # retrieve the exception


# TODO: implement _metrics_monitor_task?

# if __name__ == "__main__":
#     import os

#     import oci
#     from dotenv import load_dotenv

#     load_dotenv()

#     async def main():
#         import wave

#         # Read WAV without header
#         with wave.open("/app/oracle/original.wav", "rb") as wav_file:
#             # Get parameters
#             n_frames = wav_file.getnframes()

#             # Read raw audio data (without header)
#             audio_bytes = wav_file.readframes(n_frames)

#         config = oci.config.from_file(".oci/config", "DEFAULT")
#         config["region"] = "sa-saopaulo-1"
#         client = ai_speech.AIServiceSpeechClient(
#             config=config,
#             # TODO: review a more adequate retry strategy and timeout
#             retry_strategy=retry.NoneRetryStrategy(),
#             timeout=(10, 60),
#         )
#         session_token_details = ai_speech.models.CreateRealtimeSessionTokenDetails()
#         session_token_details.compartment_id = os.getenv("OCI_AI_COMPARTMENT_ID", "")
#         token_response = client.create_realtime_session_token(session_token_details)
#         if not token_response:
#             raise Exception("Empty token response")
#         if token_response.status != 200:
#             raise Exception("Error getting token")

#         print(token_response.data.token)

#         token = token_response.data.token

#         realtime_config: dict[str, Any] = {
#             "authenticationType": "TOKEN",
#             "token": token,
#             "compartmentId": os.getenv("OCI_AI_COMPARTMENT_ID", ""),
#         }

#         query_params: dict[str, str] = {
#             "punctuation": "NONE",
#             "languageCode": "pt",
#             "modelDomain": "GENERIC",
#             "modelType": "WHISPER",
#         }
#         base_url = "wss://realtime.aiservice.sa-saopaulo-1.oci.oraclecloud.com/ws/transcribe/stream"
#         url = f"{base_url.rstrip('/')}?{urlencode(query_params)}"
#         async with aiohttp.ClientSession() as session:
#             headers = {
#                 "User-Agent": "LiveKit Agents",
#             }
#             ws = await asyncio.wait_for(session.ws_connect(url, headers=headers), 60)
#             await ws.send_json(realtime_config)
#             connect_response = await ws.receive(timeout=5)
#             print(connect_response)
#             # await asyncio.sleep(delay=10)
#             CHUNK_SIZE = 65536
#             await asyncio.sleep(10)
#             for i in range(0, len(audio_bytes), CHUNK_SIZE):
#                 chunk = audio_bytes[i : i + CHUNK_SIZE]
#                 print(f"Chunk {i // CHUNK_SIZE + 1}: {len(chunk)} bytes")
#                 await ws.send_bytes(chunk)

#             while True:
#                 response = await ws.receive()
#                 print(json.loads(response.data)["transcriptions"][0]["transcription"])

#     asyncio.run(main())
