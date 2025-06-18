# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Extension to Elevenlabs services for improved ACE compatability."""

import base64
import json
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, calculate_word_times

from nvidia_pipecat.utils.tracing import AttachmentStrategy, traceable, traced


@traceable
class ElevenLabsTTSServiceWithEndOfSpeech(ElevenLabsTTSService):
    """ElevenLabs TTS service with end-of-speech detection.

    This class extends the base ElevenLabs TTS service to add functionality for detecting
    and handling the end of speech segments. This is useful for interactive avatar experiences
    where TTSStoppedFrames are required to signal the end of a speech segment to control lip movement
    of the avatar.

    Input frames:
        TextFrame: Text to synthesize into speech.
        TTSSpeakFrame: Alternative text input for speech synthesis.
        LLMFullResponseEndFrame: Signals LLM response completion.
        BotStoppedSpeakingFrame: Signals bot speech completion.

    Output frames:
        TTSStartedFrame: Signals TTS start.
        TTSTextFrame: Contains text being synthesized.
        TTSAudioRawFrame: Contains raw audio data.
        TTSStoppedFrame: Signals TTS completion.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the ElevenLabsTTSServiceWithEndOfSpeech.

        Shares all the parameters with the parent class ElevenLabsTTSService.

        Args:
            *args: Variable length argument list passed to parent ElevenLabsTTSService.
            **kwargs: Arbitrary keyword arguments passed to parent ElevenLabsTTSService.
        """
        super().__init__(*args, **kwargs)
        self._partial_word: dict | None = None
        self._context_id_to_close: str | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Processes frames.

        Args:
            frame (Frame): Incoming frame to process.
            direction (FrameDirection): Frame flow direction.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            self._partial_word = None

    async def flush_audio(self):
        """Flushes remaining audio in websocket connection.

        Sends special marker messages to flush audio buffer and signal end of speech.
        """
        if self._websocket and self._context_id:
            self._context_id_to_close = self._context_id
            msg = {"context_id": self._context_id, "flush": True}
            await self._websocket.send(json.dumps(msg))
            msg = {"context_id": self._context_id, "close_context": True}
            await self._websocket.send(json.dumps(msg))
            self._context_id = None

    @traced(attachment_strategy=AttachmentStrategy.NONE, name="tts")
    async def run_tts(self, text: str):
        """Run text-to-speech synthesis.

        Compared to the based class method this method is instrumented for tracing.
        """
        async for frame in super().run_tts(text):
            yield frame

    async def _receive_messages(self):
        async for message in self._get_websocket():
            msg = json.loads(message)
            # Check if this message belongs to the current context
            # The default context may return null/None for context_id
            received_ctx_id = msg.get("contextId")
            if self._context_id is not None and received_ctx_id is not None and received_ctx_id != self._context_id:
                logger.trace(f"Ignoring message from different context: {received_ctx_id}")
                continue

            if msg.get("audio"):
                await self.stop_ttfb_metrics()
                self.start_word_timestamps()

                audio = base64.b64decode(msg["audio"])
                frame = TTSAudioRawFrame(audio, self.sample_rate, 1)
                await self.push_frame(frame)
            if msg.get("alignment"):
                msg["alignment"] = self._shift_partial_words(msg["alignment"])
                word_times = calculate_word_times(msg["alignment"], self._cumulative_time)
                await self.add_word_timestamps(word_times)
                self._cumulative_time = word_times[-1][1]
            if msg.get("isFinal"):
                logger.trace(f"Received final message for context {received_ctx_id}")
                # Context has finished
                if self._context_id == received_ctx_id or self._context_id_to_close == received_ctx_id:
                    self._context_id = None
                    self._context_id_to_close = None
                    self._started = False
                    await self.push_frame(TTSStoppedFrame())

    def _shift_partial_words(self, alignment_info: dict[str, Any]) -> dict[str, Any]:
        """Shifts partial words from the previous alignment and retains incomplete words."""
        keys = ["chars", "charStartTimesMs", "charDurationsMs"]
        # Add partial word from the previous part
        if self._partial_word:
            for key in keys:
                alignment_info[key] = self._partial_word[key] + alignment_info[key]
            self._partial_word = None

        # Check if the last word is incomplete
        if not alignment_info["chars"][-1].isspace():
            # Find the last space character
            last_space_index = -1
            for i in range(len(alignment_info["chars"]) - 1, -1, -1):
                if alignment_info["chars"][i].isspace():
                    last_space_index = i + 1
                    break

            if last_space_index > -1:
                # Split into completed and partial parts
                self._partial_word = {key: alignment_info[key][last_space_index:] for key in keys}
                for key in keys:
                    alignment_info[key] = alignment_info[key][:last_space_index]

        return alignment_info
