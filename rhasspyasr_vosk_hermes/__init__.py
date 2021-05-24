"""Hermes MQTT server for Rhasspy ASR using Vosk"""
import gzip
import json
import logging
import threading
import time
import typing
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

import networkx as nx
import vosk

from rhasspyasr import Transcription, TranscriptionToken
from rhasspyhermes.asr import (
    AsrAudioCaptured,
    AsrError,
    AsrRecordingFinished,
    AsrStartListening,
    AsrStopListening,
    AsrTextCaptured,
    AsrToggleOff,
    AsrToggleOn,
    AsrToggleReason,
    AsrTrain,
    AsrTrainSuccess,
)
from rhasspyhermes.audioserver import AudioFrame, AudioSessionFrame
from rhasspyhermes.base import Message
from rhasspyhermes.client import GeneratorType, HermesClient, TopicArgs

from .utils import find_model_dir

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger("rhasspyasr_vosk_hermes")

# -----------------------------------------------------------------------------

AudioCapturedType = typing.Tuple[AsrAudioCaptured, TopicArgs]
StopListeningType = typing.Union[
    AsrRecordingFinished, AsrTextCaptured, AsrError, AudioCapturedType
]


@dataclass
class TranscriberInfo:
    """Objects for a single transcriber"""

    recognizer: typing.Optional[vosk.KaldiRecognizer] = None
    frame_queue: "Queue[typing.Optional[bytes]]" = field(default_factory=Queue)
    ready_event: threading.Event = field(default_factory=threading.Event)
    result: typing.Optional[Transcription] = None
    result_event: threading.Event = field(default_factory=threading.Event)
    result_sent: bool = False
    start_listening: typing.Optional[AsrStartListening] = None
    thread: typing.Optional[threading.Thread] = None
    audio_buffer: bytes = bytes()
    reuse: bool = True


# -----------------------------------------------------------------------------


class AsrHermesMqtt(HermesClient):
    """Hermes MQTT server for Rhasspy ASR using Vosk."""

    def __init__(
        self,
        client,
        model: typing.Union[vosk.Model, str, Path],
        words_json_path: typing.Optional[Path] = None,
        no_overwrite_train: bool = False,
        site_ids: typing.Optional[typing.List[str]] = None,
        enabled: bool = True,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        session_result_timeout: float = 20,
        reuse_transcribers: bool = True,
        lang: typing.Optional[str] = None,
    ):
        super().__init__(
            "rhasspyasr_vosk_hermes",
            client,
            site_ids=site_ids,
            sample_rate=sample_rate,
            sample_width=sample_width,
            channels=channels,
        )

        self.subscribe(
            AsrToggleOn,
            AsrToggleOff,
            AsrStartListening,
            AsrStopListening,
            AudioFrame,
            AudioSessionFrame,
            AsrTrain,
        )

        self.model: typing.Optional[vosk.Model] = None
        self.model_path: typing.Optional[Path] = None

        if isinstance(model, vosk.Model):
            # Model already loaded
            self.model = model
        else:
            # Load model later
            self.model_path = Path(model)

        self.words_json_path = words_json_path

        # True if ASR system is enabled
        self.enabled = enabled
        self.disabled_reasons: typing.Set[str] = set()

        # Seconds to wait for a result from transcriber thread
        self.session_result_timeout = session_result_timeout

        self.first_audio: bool = True

        self.lang = lang

        self.no_overwrite_train = no_overwrite_train
        self.reuse_transcribers = reuse_transcribers

        # WAV buffers for each session
        self.sessions: typing.Dict[typing.Optional[str], TranscriberInfo] = {}
        self.free_transcribers: typing.List[TranscriberInfo] = []

    # -------------------------------------------------------------------------

    async def start_listening(
        self, message: AsrStartListening
    ) -> typing.AsyncIterable[typing.Union[StopListeningType, AsrError]]:
        """Start recording audio data for a session."""
        try:
            if message.session_id in self.sessions:
                # Stop existing session
                async for stop_message in self.stop_listening(
                    AsrStopListening(
                        site_id=message.site_id, session_id=message.session_id
                    )
                ):
                    yield stop_message

            if self.free_transcribers:
                # Re-use existing transcriber
                info = self.free_transcribers.pop()

                _LOGGER.debug(
                    "Re-using existing transcriber (session_id=%s)", message.session_id
                )
            else:
                if self.model is None:
                    assert self.model_path is not None, "No model path"
                    model_dir = find_model_dir(self.model_path)
                    assert (
                        model_dir is not None
                    ), f"Vosk model not found in {self.model_path}"

                    _LOGGER.debug("Loading Vosk model from %s", model_dir)
                    self.model = vosk.Model(str(model_dir))

                assert self.model is not None

                # Create new transcriber
                recognizer_args = []
                if (
                    self.words_json_path is not None
                ) and self.words_json_path.is_file():
                    # Restrict lexicon
                    _LOGGER.debug("Loading word lists from %s", self.words_json_path)
                    with open(self.words_json_path, "r") as words_json_file:
                        recognizer_args.append(words_json_file.read())

                _LOGGER.debug("Creating new transcriber session %s", message.session_id)
                info = TranscriberInfo(
                    recognizer=vosk.KaldiRecognizer(
                        self.model, self.sample_rate, *recognizer_args
                    ),
                    reuse=self.reuse_transcribers,
                )

                def transcribe_proc(info, sample_rate, sample_width, channels):
                    try:
                        while True:
                            # Wait for session to start
                            info.ready_event.wait()
                            info.ready_event.clear()

                            start_time = time.perf_counter()
                            rec_result: typing.Optional[
                                typing.Dict[str, typing.Any]
                            ] = None
                            result: typing.Optional[Transcription] = None
                            num_samples = 0

                            # Process audio frames
                            rec = info.recognizer
                            frames = info.frame_queue.get()
                            while True:
                                if frames:
                                    # Audio frame
                                    num_samples += len(frames) // self.sample_width
                                    if rec.AcceptWaveform(frames):
                                        rec_result = json.loads(rec.Result())
                                else:
                                    # End of audio
                                    rec_result = json.loads(rec.FinalResult())

                                if rec_result is not None:
                                    # Transcription result
                                    end_time = time.perf_counter()
                                    _LOGGER.debug("Result: %s", rec_result)

                                    tokens = [
                                        TranscriptionToken(
                                            token=word.get("word", ""),
                                            start_time=float(word.get("start", 0.0)),
                                            end_time=float(word.get("end", 0.0)),
                                            likelihood=float(word.get("conf", 1.0)),
                                        )
                                        for word in rec_result.get("result", [])
                                    ]
                                    likelihood = 1.0

                                    for token in tokens:
                                        likelihood = min(likelihood, token.likelihood)

                                    result = Transcription(
                                        text=rec_result.get("text", ""),
                                        likelihood=likelihood,
                                        tokens=tokens,
                                        wav_seconds=num_samples / self.sample_rate,
                                        transcribe_seconds=(end_time - start_time),
                                    )
                                    break

                                if not frames:
                                    break

                                frames = info.frame_queue.get()

                            _LOGGER.debug("Transcription result: %s", result)

                            assert (
                                result is not None and result.text
                            ), "Null transcription"

                            # Signal completion
                            info.result = result
                            info.result_event.set()
                    except Exception:
                        _LOGGER.exception("session proc")

                        # Mark as not reusable
                        info.reuse = False

                        # Signal failure
                        info.recognizer = None
                        info.result = Transcription(
                            text="", likelihood=0, transcribe_seconds=0, wav_seconds=0
                        )
                        info.result_event.set()

                # Run in separate thread
                info.thread = threading.Thread(
                    target=transcribe_proc,
                    args=(info, self.sample_rate, self.sample_width, self.channels),
                    daemon=True,
                )

                info.thread.start()

            # ---------------------------------------------------------------------

            # Settings for session
            info.start_listening = message

            # Signal session thread to start
            info.ready_event.set()

            self.sessions[message.session_id] = info
            _LOGGER.debug("Starting listening (session_id=%s)", message.session_id)
            self.first_audio = True
        except Exception as e:
            _LOGGER.exception("start_listening")
            yield AsrError(
                error=str(e),
                context=repr(message),
                site_id=message.site_id,
                session_id=message.session_id,
            )

    async def stop_listening(
        self, message: AsrStopListening
    ) -> typing.AsyncIterable[StopListeningType]:
        """Stop recording audio data for a session."""
        info = self.sessions.pop(message.session_id, None)
        if info:
            try:
                # Trigger publishing of transcription on end of session
                async for result in self.finish_session(
                    info, message.site_id, message.session_id
                ):
                    yield result

                if info.reuse and (info.recognizer is not None):
                    # Reset state
                    info.result = None
                    info.result_event.clear()
                    info.result_sent = False
                    info.result = None
                    info.start_listening = None
                    info.audio_buffer = bytes()

                    while info.frame_queue.qsize() > 0:
                        info.frame_queue.get_nowait()

                    # Add to free pool
                    self.free_transcribers.append(info)
            except Exception as e:
                _LOGGER.exception("stop_listening")
                yield AsrError(
                    error=str(e),
                    context=repr(info.recognizer),
                    site_id=message.site_id,
                    session_id=message.session_id,
                )

        _LOGGER.debug("Stopping listening (session_id=%s)", message.session_id)

    async def handle_audio_frame(
        self,
        frame_wav_bytes: bytes,
        site_id: str = "default",
        session_id: typing.Optional[str] = None,
    ) -> typing.AsyncIterable[
        typing.Union[
            AsrRecordingFinished,
            AsrTextCaptured,
            AsrError,
            typing.Tuple[AsrAudioCaptured, TopicArgs],
        ]
    ]:
        """Process single frame of WAV audio"""

        # Don't process audio if no sessions
        if not self.sessions:
            return

        audio_data = self.maybe_convert_wav(frame_wav_bytes)

        if session_id is None:
            # Add to every open session
            target_sessions = list(self.sessions.items())
        else:
            # Add to single session
            target_sessions = [(session_id, self.sessions[session_id])]

        # Add to every open session with matching site_id
        for target_id, info in target_sessions:
            try:
                assert info.start_listening is not None

                # Match site_id
                if info.start_listening.site_id != site_id:
                    continue

                if info.result:
                    # Trigger publishing of transcription on end of session
                    async for result in self.finish_session(
                        info, site_id, info.start_listening.session_id
                    ):
                        yield result
                else:
                    # Push to transcription thread
                    info.frame_queue.put(audio_data)

                    # Add to session audio buffer
                    info.audio_buffer += audio_data
            except Exception as e:
                _LOGGER.exception("handle_audio_frame")
                yield AsrError(
                    error=str(e),
                    context=repr(info.recognizer),
                    site_id=site_id,
                    session_id=target_id,
                )

    async def finish_session(
        self, info: TranscriberInfo, site_id: str, session_id: typing.Optional[str]
    ) -> typing.AsyncIterable[
        typing.Union[AsrRecordingFinished, AsrTextCaptured, AudioCapturedType]
    ]:
        """Publish transcription result for a session if not already published"""

        audio_data = info.audio_buffer

        if not info.result_sent:
            # Send recording finished message
            yield AsrRecordingFinished(site_id=site_id, session_id=session_id)

            # Avoid re-sending transcription
            info.result_sent = True

            # Last chunk
            info.frame_queue.put(None)

            # Wait for result
            result_success = info.result_event.wait(timeout=self.session_result_timeout)
            if not result_success:
                # Mark transcription as non-reusable
                info.reuse = False

            transcription = info.result
            assert info.start_listening is not None

            if transcription:
                # Successful transcription
                yield (
                    AsrTextCaptured(
                        text=transcription.text,
                        likelihood=transcription.likelihood,
                        seconds=transcription.transcribe_seconds,
                        site_id=site_id,
                        session_id=session_id,
                        lang=(info.start_listening.lang or self.lang),
                    )
                )
            else:
                # Empty transcription
                yield AsrTextCaptured(
                    text="",
                    likelihood=0,
                    seconds=0,
                    site_id=site_id,
                    session_id=session_id,
                    lang=(info.start_listening.lang or self.lang),
                )

            if info.start_listening.send_audio_captured:
                wav_bytes = self.to_wav_bytes(audio_data)

                # Send audio data
                yield (
                    # pylint: disable=E1121
                    AsrAudioCaptured(wav_bytes),
                    {"site_id": site_id, "session_id": session_id},
                )

    # -------------------------------------------------------------------------

    async def handle_train(
        self, train: AsrTrain, site_id: str = "default"
    ) -> typing.AsyncIterable[
        typing.Union[typing.Tuple[AsrTrainSuccess, TopicArgs], AsrError]
    ]:
        """Re-trains ASR system"""
        try:
            if not self.no_overwrite_train and self.words_json_path:
                _LOGGER.debug("Loading %s", train.graph_path)
                with gzip.GzipFile(train.graph_path, mode="rb") as graph_gzip:
                    graph = nx.readwrite.gpickle.read_gpickle(graph_gzip)

                words = set()
                for _, _, edge_data in graph.edges(data=True):
                    word = edge_data.get("olabel", "")
                    if word and (not word.startswith("__")):
                        words.add(word)

                # Generate list of all words used
                with open(self.words_json_path, "w") as words_json_file:
                    json.dump([" ".join(words), "[unk]"], words_json_file)
            else:
                _LOGGER.warning("Not overwriting word lsits")

            # Clear out existing transcribers so models can reload on next voice command
            self.free_transcribers = []
            for info in self.sessions.values():
                info.reuse = False

            yield (AsrTrainSuccess(id=train.id), {"site_id": site_id})
        except Exception as e:
            _LOGGER.exception("handle_train")
            yield AsrError(
                error=str(e),
                context="handle_train",
                site_id=site_id,
                session_id=train.id,
            )

    # -------------------------------------------------------------------------

    async def on_message(
        self,
        message: Message,
        site_id: typing.Optional[str] = None,
        session_id: typing.Optional[str] = None,
        topic: typing.Optional[str] = None,
    ) -> GeneratorType:
        """Received message from MQTT broker."""
        # Check enable/disable messages
        if isinstance(message, AsrToggleOn):
            if message.reason == AsrToggleReason.UNKNOWN:
                # Always enable on unknown
                self.disabled_reasons.clear()
            else:
                self.disabled_reasons.discard(message.reason)

            if self.disabled_reasons:
                _LOGGER.debug("Still disabled: %s", self.disabled_reasons)
            else:
                self.enabled = True
                self.first_audio = True
                _LOGGER.debug("Enabled")
        elif isinstance(message, AsrToggleOff):
            self.enabled = False
            self.disabled_reasons.add(message.reason)
            _LOGGER.debug("Disabled")
        elif isinstance(message, AudioFrame):
            if self.enabled:
                assert site_id, "Missing site_id"
                if self.first_audio:
                    _LOGGER.debug("Receiving audio")
                    self.first_audio = False

                # Add to all active sessions
                async for frame_result in self.handle_audio_frame(
                    message.wav_bytes, site_id=site_id
                ):
                    yield frame_result
        elif isinstance(message, AudioSessionFrame):
            if self.enabled:
                assert site_id and session_id, "Missing site_id or session_id"
                if session_id in self.sessions:
                    if self.first_audio:
                        _LOGGER.debug("Receiving audio")
                        self.first_audio = False

                    # Add to specific session only
                    async for session_frame_result in self.handle_audio_frame(
                        message.wav_bytes, site_id=site_id, session_id=session_id
                    ):
                        yield session_frame_result
        elif isinstance(message, AsrStartListening):
            # hermes/asr/startListening
            async for start_result in self.start_listening(message):
                yield start_result
        elif isinstance(message, AsrStopListening):
            # hermes/asr/stopListening
            async for stop_result in self.stop_listening(message):
                yield stop_result
        elif isinstance(message, AsrTrain):
            # rhasspy/asr/<site_id>/train
            assert site_id, "Missing site_id"
            async for train_result in self.handle_train(message, site_id=site_id):
                yield train_result
        else:
            _LOGGER.warning("Unexpected message: %s", message)
