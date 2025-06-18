# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""NVIDIA Riva speech services implementation.

This module provides integration with NVIDIA Riva's speech services, including:
- Text-to-Speech (TTS) with support for multiple voices and languages
- Automatic Speech Recognition (ASR) with streaming capabilities

The services can be configured to use either a local Riva Speech Server or
NVIDIA's cloud-hosted models through NVCF.

For documentation on how to configure the Riva Speech models, please refer to the
[Riva Speech Quick Start Guide](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html).
"""

import asyncio
import concurrent.futures
from collections.abc import AsyncGenerator

import riva.client
from loguru import logger
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from riva.client.proto.riva_audio_pb2 import AudioEncoding

from nvidia_pipecat.frames.riva import RivaInterimTranscriptionFrame
from nvidia_pipecat.utils.tracing import AttachmentStrategy, traceable, traced


@traceable
class RivaTTSService(TTSService):
    """NVIDIA Riva Text-to-Speech service implementation.

    Provides speech synthesis using NVIDIA's Riva TTS models with support for
    multiple voices, languages, and custom dictionaries.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        server: str = "grpc.nvcf.nvidia.com:443",
        voice_id: str = "English-US.Female-1",
        sample_rate: int = 16000,
        function_id: str = "0149dedb-2be8-4195-b9a0-e57e0e14f972",
        language: Language | None = Language.EN_US,
        zero_shot_quality: int | None = 20,
        model: str = "fastpitch-hifigan-tts",
        custom_dictionary: dict | None = None,
        encoding: AudioEncoding = AudioEncoding.LINEAR_PCM,
        zero_shot_audio_prompt_file: str | None = None,
        audio_prompt_encoding: AudioEncoding = AudioEncoding.LINEAR_PCM,
        use_ssl: bool = False,
        **kwargs,
    ):
        """Initializes the Riva TTS service.

        Args:
            api_key (str | None, optional): API key for authentication. Defaults to None.
            server (str, optional): Server address for Riva service. Defaults to "grpc.nvcf.nvidia.com:443".
            voice_id (str, optional): Voice identifier. Defaults to "English-US.Female-1".
            sample_rate (int, optional): Audio sample rate in Hz. Defaults to 16000.
            function_id (str, optional): Function identifier for the service.
                Defaults to "0149dedb-2be8-4195-b9a0-e57e0e14f972".
            language (Language | None, optional): Language for synthesis. Defaults to Language.EN_US.
            zero_shot_quality (int | None, optional): Quality level for synthesis. Defaults to 20.
            model (str, optional): Model name for synthesis. Defaults to "fastpitch-hifigan-tts".
            custom_dictionary (dict | None, optional): Custom pronunciation dictionary. Defaults to None.
            encoding (AudioEncoding, optional): Audio encoding format. Defaults to AudioEncoding.LINEAR_PCM.
            zero_shot_audio_prompt_file (str | None, optional): Path to audio prompt file. Defaults to None.
            audio_prompt_encoding (AudioEncoding, optional): Encoding of audio prompt.
                Defaults to AudioEncoding.LINEAR_PCM.
            use_ssl (bool, optional): Whether to use SSL for connection. Defaults to False.
            **kwargs: Additional keyword arguments passed to parent class.

        Raises:
            Exception: If required modules are missing or connection fails.

        Usage:
            If server is not set then it defaults to "grpc.nvcf.nvidia.com:443" and use NVCF hosted models.
            Update function ID to use a different NVCF model. API key is required for NVCF hosted models.
            For using locally deployed Riva Speech Server, set server to "localhost:50051" and
            follow the quick start guide to setup the server.
        """
        super().__init__(
            sample_rate=sample_rate,
            push_text_frames=False,
            push_stop_frames=True,
            **kwargs,
        )
        self._api_key = api_key
        self._function_id = function_id
        self._voice_id = voice_id
        self._sample_rate = sample_rate
        self._language_code = language
        self._zero_shot_quality = zero_shot_quality
        self.set_model_name(model)
        self.set_voice(voice_id)
        self._custom_dictionary = custom_dictionary
        self._encoding = encoding
        self._zero_shot_audio_prompt_file = zero_shot_audio_prompt_file
        self._audio_prompt_encoding = audio_prompt_encoding

        metadata = [
            ["function-id", function_id],
            ["authorization", f"Bearer {api_key}"],
        ]

        if server == "grpc.nvcf.nvidia.com:443":
            use_ssl = True

        try:
            auth = riva.client.Auth(None, use_ssl, server, metadata)
            self._service = riva.client.SpeechSynthesisService(auth)
            # warm up the service
            _ = self._service.stub.GetRivaSynthesisConfig(riva.client.proto.riva_tts_pb2.RivaSynthesisConfigRequest())
        except Exception as e:
            logger.error(
                "In order to use nvidia Riva TTSService or STTService, you will either need a locally "
                "deployed Riva Speech Server with ASR and TTS models (Follow riva quick start guide at "
                "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html and "
                "edit the config file to deploy which model you want to use and set the server url to "
                "localhost:50051), or you can set the NVIDIA_API_KEY environment "
                "variable to connect with nvcf hosted models."
            )
            raise Exception(f"Missing module: {e}") from e

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            bool: True as this service supports metric generation.
        """
        return True

    async def _push_tts_frames(self, text: str):
        """Override base class method to push text frames immediately."""
        # Remove leading newlines only
        text = text.lstrip("\n")

        # Don't send only whitespace. This causes problems for some TTS models. But also don't
        # strip all whitespace, as whitespace can influence prosody.
        if not text.strip():
            return

        # This is just a flag that indicates if we sent something to the TTS
        # service. It will be cleared if we sent text because of a TTSSpeakFrame
        # or when we received an LLMFullResponseEndFrame
        self._processing_text = True

        await self.start_processing_metrics()
        # Process all filter.
        for filter in self._text_filters:
            filter.reset_interruption()
            text = filter.filter(text)

        if text:
            await self.process_generator(self.run_tts(text))
        await self.stop_processing_metrics()

    @traced(attachment_strategy=AttachmentStrategy.NONE, name="tts")
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Run text-to-speech synthesis."""
        logger.debug(f"Generating TTS: [{text}]")
        responses = self._service.synthesize_online(
            text,
            self._voice_id,
            self._language_code,
            sample_rate_hz=self._sample_rate,
            zero_shot_audio_prompt_file=self._zero_shot_audio_prompt_file,
            audio_prompt_encoding=self._audio_prompt_encoding,
            zero_shot_quality=self._zero_shot_quality,
            custom_dictionary=self._custom_dictionary,
            encoding=self._encoding,
        )

        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        # Push text frame immediately after TTSStartedFrame.
        # TTSService base processor will push the tts text after sending generated tts audio downstream
        # Need to push the text before audio frame for better TTS transcription.
        yield TTSTextFrame(text)

        async def get_next_response(iterator):
            def _next():
                try:
                    return next(iterator)
                except StopIteration:
                    return None

            return await asyncio.get_event_loop().run_in_executor(None, _next)

        response_iterator = iter(responses)

        while (resp := await get_next_response(response_iterator)) is not None:
            try:
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(
                    audio=resp.audio,
                    sample_rate=self._sample_rate,
                    num_channels=1,
                )
                yield frame
            except Exception as e:
                logger.error(f"{self} Error processing TTS response: {e}")
                break

        await self.start_tts_usage_metrics(text)
        yield TTSStoppedFrame()


@traceable
class RivaASRService(STTService):
    """NVIDIA Riva Automatic Speech Recognition service.

    Provides streaming speech recognition using Riva ASR models with support for:
        - Real-time transcription
        - Interim results
        - Interruption handling
        - Voice activity detection
        - Language model customization
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        server: str = "grpc.nvcf.nvidia.com:443",
        function_id: str = "1598d209-5e27-4d3c-8079-4751568b1081",
        language: Language | None = Language.EN_US,
        model: str = "parakeet-1.1b-en-US-asr-streaming-asr-bls-ensemble",
        profanity_filter: bool = False,
        automatic_punctuation: bool = False,
        no_verbatim_transcripts: bool = False,
        boosted_lm_words: dict | None = None,
        boosted_lm_score: float = 4.0,
        start_history: int = -1,
        start_threshold: float = -1.0,
        stop_history: int = 500,
        stop_threshold: float = -1.0,
        stop_history_eou: int = 240,
        stop_threshold_eou: float = -1.0,
        custom_configuration: str = "enable_vad_endpointing:true",
        sample_rate: int = 16000,
        audio_channel_count: int = 1,
        max_alternatives: int = 1,
        interim_results: bool = True,
        generate_interruptions: bool = False,  # Only set to True if transport VAD is disabled
        idle_timeout: int = 30,  # Timeout for idle Riva ASR request
        use_ssl: bool = False,
        **kwargs,
    ):
        """Initializes the Riva ASR service.

        Args:
            api_key: NVIDIA API key for cloud access.
            server: Riva server address.
            function_id: NVCF function identifier.
            language: Language for recognition.
            model: ASR model name.
            profanity_filter: Enable profanity filtering.
            automatic_punctuation: Enable automatic punctuation.
            no_verbatim_transcripts: Disable verbatim transcripts.
            boosted_lm_words: Words to boost in language model.
            boosted_lm_score: Score for boosted words.
            start_history: VAD start history frames.
            start_threshold: VAD start threshold.
            stop_history: VAD stop history frames.
            stop_threshold: VAD stop threshold.
            stop_history_eou: End-of-utterance history frames.
            stop_threshold_eou: End-of-utterance threshold.
            custom_configuration: Additional configuration string.
            sample_rate: Audio sample rate in Hz.
            audio_channel_count: Number of audio channels.
            max_alternatives: Maximum number of alternatives.
            interim_results: Enable interim results.
            generate_interruptions: Enable interruption events.
            idle_timeout: Timeout for idle ASR request in seconds.
            use_ssl: Enable SSL connection.
            **kwargs: Additional arguments for STTService.

        Usage:
            If server is not set then it defaults to "grpc.nvcf.nvidia.com:443" and use NVCF hosted models.
            Update function ID to use a different NVCF model. API key is required for NVCF hosted models.
            For using locally deployed Riva Speech Server, set server to "localhost:50051" and
            follow the quick start guide to setup the server.
        """
        super().__init__(**kwargs)
        self._profanity_filter = profanity_filter
        self._automatic_punctuation = automatic_punctuation
        self._no_verbatim_transcripts = no_verbatim_transcripts
        self._language_code = language
        self._boosted_lm_words = boosted_lm_words
        self._boosted_lm_score = boosted_lm_score
        self._start_history = start_history
        self._start_threshold = start_threshold
        self._stop_history = stop_history
        self._stop_threshold = stop_threshold
        self._stop_history_eou = stop_history_eou
        self._stop_threshold_eou = stop_threshold_eou
        self._custom_configuration = custom_configuration
        self._sample_rate: int = sample_rate
        self._model = model
        self._audio_channel_count = audio_channel_count
        self._max_alternatives = max_alternatives
        self._interim_results = interim_results
        self._idle_timeout = idle_timeout
        self.last_transcript_frame = None
        self.set_model_name(model)

        metadata = [
            ["function-id", function_id],
            ["authorization", f"Bearer {api_key}"],
        ]

        if server == "grpc.nvcf.nvidia.com:443":
            use_ssl = True

        try:
            auth = riva.client.Auth(None, use_ssl, server, metadata)
            self._asr_service = riva.client.ASRService(auth)
        except Exception as e:
            logger.error(
                "In order to use nvidia Riva TTSService or STTService, you will either need a locally "
                "deployed Riva Speech Server with ASR and TTS models (Follow riva quick start guide at "
                "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html and "
                "edit the config file to deploy which model you want to use and set the server url to "
                "localhost:50051), or you can set the NVIDIA_API_KEY environment "
                "variable to connect with nvcf hosted models."
            )
            raise Exception(f"Missing module: {e}") from e

        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=self._language_code,
                model=self._model,
                max_alternatives=self._max_alternatives,
                profanity_filter=self._profanity_filter,
                enable_automatic_punctuation=self._automatic_punctuation,
                verbatim_transcripts=not self._no_verbatim_transcripts,
                sample_rate_hertz=self._sample_rate,
                audio_channel_count=self._audio_channel_count,
            ),
            interim_results=self._interim_results,
        )
        riva.client.add_word_boosting_to_config(config, self._boosted_lm_words, self._boosted_lm_score)
        riva.client.add_endpoint_parameters_to_config(
            config,
            self._start_history,
            self._start_threshold,
            self._stop_history,
            self._stop_history_eou,
            self._stop_threshold,
            self._stop_threshold_eou,
        )
        riva.client.add_custom_configuration_to_config(config, self._custom_configuration)
        self._config = config

        self._queue = asyncio.Queue()
        self._generate_interruptions = generate_interruptions
        if self._generate_interruptions:
            self._vad_state = VADState.QUIET

        # Initialize the thread task and response task
        self._thread_task = None
        self._response_task = None

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            bool: False as this service does not support metric generation.
        """
        return False

    async def start(self, frame: StartFrame):
        """Start the ASR service.

        Args:
            frame: The StartFrame that triggered the start.
        """
        await super().start(frame)
        self._response_task = self.create_task(self._response_task_handler())
        self._response_queue = asyncio.Queue()

    async def stop(self, frame: EndFrame):
        """Stop the ASR service and cleanup resources.

        Args:
            frame: The EndFrame that triggered the stop.
        """
        await super().stop(frame)
        await self._stop_tasks()

    async def cancel(self, frame: CancelFrame):
        """Cancel the ASR service and cleanup resources.

        Args:
            frame: The CancelFrame that triggered the cancellation.
        """
        await super().cancel(frame)
        await self._stop_tasks()

    async def _stop_tasks(self):
        if self._thread_task is not None and not self._thread_task.done():
            await self.cancel_task(self._thread_task)
        if self._response_task is not None and not self._response_task.done():
            await self.cancel_task(self._response_task)

    def _response_handler(self):
        try:
            logger.debug("Sending new Riva ASR streaming request...")
            responses = self._asr_service.streaming_response_generator(
                audio_chunks=self,
                streaming_config=self._config,
            )
            for response in responses:
                if not response.results:
                    continue
                asyncio.run_coroutine_threadsafe(self._response_queue.put(response), self.get_event_loop())
        except Exception as e:
            logger.error(f"Error in Riva ASR stream: {e}")
            raise
        logger.debug("Riva ASR streaming request terminated.")

    @traced(attachment_strategy=AttachmentStrategy.NONE, name="asr")
    async def _thread_task_handler(self):
        try:
            self._thread_running = True
            await asyncio.to_thread(self._response_handler)
        except asyncio.CancelledError:
            self._thread_running = False
            raise

    async def _handle_interruptions(self, frame: Frame):
        if self.interruptions_allowed:
            # Make sure we notify about interruptions quickly out-of-band.
            if isinstance(frame, UserStartedSpeakingFrame):
                logger.debug("User started speaking")
                await self._start_interruption()
                # Push an out-of-band frame (i.e. not using the ordered push
                # frame task) to stop everything, specially at the output
                # transport.
                await self.push_frame(StartInterruptionFrame())
            elif isinstance(frame, UserStoppedSpeakingFrame):
                logger.debug("User stopped speaking")
                await self._stop_interruption()
                await self.push_frame(StopInterruptionFrame())

        await self.push_frame(frame)

    async def _handle_response(self, response):
        """Process ASR response and generate appropriate transcription frames.

        Handles three types of transcription results:
        1. Final results (is_final=True): Complete, confirmed transcriptions
        2. Stable interim results (stability=1.0): High-confidence partial results
        3. Partial results (stability<1.0): Lower-confidence, in-progress transcriptions

        Also manages voice activity detection (VAD) state and interruption handling
        when enabled. Each type of result generates appropriate transcription frames
        with different stability values.
        """
        partial_transcript = ""
        for result in response.results:
            if result and not result.alternatives:
                continue
            transcript = result.alternatives[0].transcript
            if transcript and len(transcript) > 0:
                await self.stop_ttfb_metrics()
                if result.is_final:
                    await self.stop_processing_metrics()
                    if self._generate_interruptions:
                        self._vad_state = VADState.QUIET
                        await self._handle_interruptions(UserStoppedSpeakingFrame())
                    logger.debug(f"Final user transcript: [{transcript}]")
                    await self.push_frame(TranscriptionFrame(transcript, "", time_now_iso8601(), None))
                    self.last_transcript_frame = None
                    break
                elif result.stability == 1.0:
                    if self._generate_interruptions and self._vad_state != VADState.SPEAKING:
                        self._vad_state = VADState.SPEAKING
                        await self._handle_interruptions(UserStartedSpeakingFrame())
                    if (
                        self.last_transcript_frame is None
                        or (self.last_transcript_frame.stability != 1.0)
                        or (self.last_transcript_frame.text.rstrip() != transcript.rstrip())
                    ):
                        logger.debug(f"Interim user transcript: [{transcript}]")
                        frame = RivaInterimTranscriptionFrame(
                            transcript, "", time_now_iso8601(), None, stability=result.stability
                        )
                        await self.push_frame(frame)
                        self.last_transcript_frame = frame
                    break
                else:
                    if self._generate_interruptions and self._vad_state != VADState.SPEAKING:
                        self._vad_state = VADState.SPEAKING
                        await self._handle_interruptions(UserStartedSpeakingFrame())
                    partial_transcript += transcript

        if len(partial_transcript) > 0 and (
            self.last_transcript_frame is None
            or (self.last_transcript_frame.stability == 1.0)
            or (self.last_transcript_frame.text.rstrip() != partial_transcript.rstrip())
        ):
            logger.debug(f"Partial user transcript: [{partial_transcript}]")
            frame = RivaInterimTranscriptionFrame(partial_transcript, "", time_now_iso8601(), None, stability=0.1)
            await self.push_frame(frame)
            self.last_transcript_frame = frame

    async def _response_task_handler(self):
        while True:
            try:
                response = await self._response_queue.get()
                await self._handle_response(response)
            except asyncio.CancelledError:
                break

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Run speech-to-text recognition.

        Args:
            audio: The audio data to process.

        Yields:
            Frame: A sequence of frames containing the recognition results.
        """
        if self._thread_task is None or self._thread_task.done():
            self._thread_task = self.create_task(self._thread_task_handler())
        await self._queue.put(audio)
        yield None

    def __next__(self) -> bytes:
        """Get the next audio chunk for processing.

        Returns:
            bytes: The next audio chunk.

        Raises:
            StopIteration: When no more audio chunks are available.
        """
        if not self._thread_running:
            raise StopIteration
        try:
            future = asyncio.run_coroutine_threadsafe(self._queue.get(), self.get_event_loop())
            result = future.result(timeout=self._idle_timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.info(f"ASR service is idle for {self._idle_timeout} seconds, terminating active RIVA ASR request...")
            self._thread_task = None
            raise StopIteration from None
        except Exception as e:
            future.cancel()
            raise e
        return result

    def __iter__(self):
        """Get iterator for audio chunks.

        Returns:
            RivaASRService: Self reference for iteration.
        """
        return self
