# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the ElevenLabsTTSServiceWithEndOfSpeech class."""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, patch

import pytest
from loguru import logger
from pipecat.frames.frames import TTSAudioRawFrame, TTSSpeakFrame, TTSStoppedFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from websockets.protocol import State

from nvidia_pipecat.services.elevenlabs import ElevenLabsTTSServiceWithEndOfSpeech
from nvidia_pipecat.utils.logging import setup_default_ace_logging
from tests.unit.utils import FrameStorage, ignore_ids, run_interactive_test

setup_default_ace_logging(level="TRACE")


class MockWebSocket:
    """Mock WebSocket for testing ElevenLabs service.

    Attributes:
        messages_to_return: List of messages to return during testing.
        sent_messages: List of messages sent through the socket.
        state: Current WebSocket connection state.
        close_rcvd: Close frame received flag.
        close_rcvd_then_sent: Close frame received and sent flag.
        close_sent: Close frame sent flag.
        closed: Whether the WebSocket is closed.
    """

    def __init__(self, messages_to_return):
        """Initialize MockWebSocket.

        Args:
            messages_to_return (list): List of messages to return during testing.
        """
        self.messages_to_return = messages_to_return
        self.sent_messages = []
        self.state = State.OPEN
        self.close_rcvd = None
        self.close_rcvd_then_sent = None
        self.close_sent = None
        self.closed = False

    async def send(self, message: str) -> None:
        """Sends a message through the mock socket.

        Args:
            message (str): Message to send.
        """
        self.sent_messages.append(json.loads(message))

    async def ping(self) -> bool:
        """Simulates WebSocket heartbeat.

        Returns:
            bool: Always True for testing.
        """
        return True

    async def close(self) -> bool:
        """Closes the mock WebSocket connection.

        Returns:
            bool: Always True for testing.
        """
        self.state = State.CLOSED
        self.closed = True
        return True

    async def __aiter__(self):
        """Async iterator for messages.

        Yields:
            str: JSON-encoded message.
        """
        for msg in self.messages_to_return:
            yield json.dumps(msg)
        while self.state != State.CLOSED:
            await asyncio.sleep(1.0)
            yield "{}"


@pytest.mark.asyncio()
async def test_elevenlabs_tts_service_with_end_of_speech():
    """Test ElevenLabsTTSServiceWithEndOfSpeech functionality.

    Tests:
        - End-of-speech boundary marker handling
        - Audio message processing
        - Alignment message processing
        - TTSStoppedFrame generation

    Raises:
        AssertionError: If frame processing or timing is incorrect.
    """
    # Test audio data
    test_audio = b"test_audio_data"
    test_audio_b64 = base64.b64encode(test_audio).decode()

    # Test cases with different message sequences
    testcases = {
        "Normal audio with boundary marker": {
            "frames_to_send": [TTSSpeakFrame("Hello")],
            "messages": [
                {
                    "audio": test_audio_b64,
                    "alignment": {
                        "chars": ["H", "e", "l", "l", "o", "\u200b"],
                        "charStartTimesMs": [0, 3, 7, 9, 11, 12],
                        "charDurationsMs": [3, 4, 2, 2, 1, 1],
                    },
                },
            ],
            "expected_frames": [
                TTSAudioRawFrame(test_audio, 16000, 1),
                TTSStoppedFrame(),
            ],
        },
        "Multiple audio chunks": {
            "frames_to_send": [TTSSpeakFrame("Test")],
            "messages": [
                {
                    "audio": test_audio_b64,
                    "alignment": {
                        "chars": ["T", "e"],
                        "charStartTimesMs": [0, 3],
                        "charDurationsMs": [3, 4],
                    },
                },
                {
                    "audio": test_audio_b64,
                    "alignment": {
                        "chars": ["s", "t", "\u200b"],
                        "charStartTimesMs": [7, 9, 11],
                        "charDurationsMs": [2, 2, 1],
                    },
                },
            ],
            "expected_frames": [
                TTSAudioRawFrame(test_audio, 16000, 1),
                TTSAudioRawFrame(test_audio, 16000, 1),
                TTSStoppedFrame(),
            ],
        },
    }

    for tc_name, tc_data in testcases.items():
        logger.info(f"Verifying test case: {tc_name}")

        # Create mock websocket with test messages
        mock_websocket = MockWebSocket(tc_data["messages"])

        # mock = AsyncMock()
        # mock.return_value = mock_websocket

        with patch("pipecat.services.elevenlabs.tts.websockets.connect", new=AsyncMock()) as mock:
            mock.return_value = mock_websocket
            tts_service = ElevenLabsTTSServiceWithEndOfSpeech(
                api_key="test_api_key", voice_id="test_voice_id", sample_rate=16000, channels=1
            )

            storage = FrameStorage()
            pipeline = Pipeline([tts_service, storage])

            async def test_routine(task: PipelineTask, test_data=tc_data, s=storage):
                for frame in test_data["frames_to_send"]:
                    await task.queue_frame(frame)
                # Wait for all expected frames
                for expected_frame in test_data["expected_frames"]:
                    await s.wait_for_frame(ignore_ids(expected_frame))
                    print(f"got frame to be sent {expected_frame}")

                # TODO: investigate why we need to cancel here
                await task.cancel()

            await run_interactive_test(pipeline, test_coroutine=test_routine)
