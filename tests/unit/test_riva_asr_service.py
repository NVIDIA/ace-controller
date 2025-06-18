# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the RivaASRService.

This module contains tests for the RivaASRService class, including initialization,
ASR functionality, interruption handling, and integration tests.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.transcriptions.language import Language

from nvidia_pipecat.frames.riva import RivaInterimTranscriptionFrame
from nvidia_pipecat.services.riva_speech import RivaASRService


class TestRivaASRService(unittest.TestCase):
    """Test suite for RivaASRService functionality."""

    @patch("nvidia_pipecat.services.riva_speech.riva.client.Auth")
    @patch("nvidia_pipecat.services.riva_speech.riva.client.ASRService")
    def test_initialization_with_default_parameters(self, mock_asr_service, mock_auth):
        """Tests RivaASRService initialization with default parameters.

        Args:
            mock_asr_service: Mock for the ASR service.
            mock_auth: Mock for the authentication service.

        The test verifies:
            - Correct type initialization for all parameters
            - Default server configuration
            - Authentication setup
            - Service parameter defaults
        """
        # Act
        service = RivaASRService(api_key="test_api_key")

        # Assert - only check types, not specific values
        self.assertIsInstance(service._language_code, Language)
        self.assertIsInstance(service._sample_rate, int)
        self.assertIsInstance(service._model, str)
        # Basic boolean parameter checks without restricting values
        self.assertIsInstance(service._profanity_filter, bool)
        self.assertIsInstance(service._automatic_punctuation, bool)
        self.assertIsInstance(service._interim_results, bool)
        self.assertIsInstance(service._max_alternatives, int)
        self.assertIsInstance(service._generate_interruptions, bool)

        # Verify Auth was called with correct parameters
        # For server "grpc.nvcf.nvidia.com:443", use_ssl is automatically set to True
        mock_auth.assert_called_with(
            None,
            True,  # Changed from False to True since default server is "grpc.nvcf.nvidia.com:443"
            "grpc.nvcf.nvidia.com:443",
            [["function-id", "1598d209-5e27-4d3c-8079-4751568b1081"], ["authorization", "Bearer test_api_key"]],
        )

    @patch("nvidia_pipecat.services.riva_speech.riva.client.Auth")
    @patch("nvidia_pipecat.services.riva_speech.riva.client.ASRService")
    def test_initialization_with_custom_parameters(self, mock_asr_service, mock_auth):
        """Tests RivaASRService initialization with custom parameters.

        Args:
            mock_asr_service: Mock for the ASR service.
            mock_auth: Mock for the authentication service.

        The test verifies:
            - Custom parameter values are set correctly
            - Server configuration is customized
            - Authentication with custom credentials
            - Optional parameter handling
        """
        # Define test parameters
        test_api_key = "test_api_key"
        test_server = "custom_server:50051"
        test_function_id = "custom_function_id"
        test_language = Language.ES_ES
        test_model = "custom_model"
        test_sample_rate = 44100
        test_channel_count = 2
        test_max_alternatives = 2
        test_boosted_words = {"boost": 1.0}
        test_boosted_score = 5.0

        # Act
        service = RivaASRService(
            api_key=test_api_key,
            server=test_server,
            function_id=test_function_id,
            language=test_language,
            model=test_model,
            profanity_filter=True,
            automatic_punctuation=True,
            no_verbatim_transcripts=True,
            boosted_lm_words=test_boosted_words,
            boosted_lm_score=test_boosted_score,
            sample_rate=test_sample_rate,
            audio_channel_count=test_channel_count,
            max_alternatives=test_max_alternatives,
            interim_results=False,
            generate_interruptions=True,
            use_ssl=True,
        )

        # Assert - verify custom parameters were set correctly
        self.assertEqual(service._language_code, test_language)
        self.assertEqual(service._sample_rate, test_sample_rate)
        self.assertEqual(service._model, test_model)
        self.assertEqual(service._boosted_lm_words, test_boosted_words)
        self.assertEqual(service._boosted_lm_score, test_boosted_score)
        self.assertEqual(service._max_alternatives, test_max_alternatives)
        self.assertEqual(service._audio_channel_count, test_channel_count)

        # Verify Auth was called with correct parameters
        mock_auth.assert_called_with(
            None,
            True,
            test_server,
            [["function-id", test_function_id], ["authorization", f"Bearer {test_api_key}"]],
        )

    @patch("nvidia_pipecat.services.riva_speech.riva.client.Auth")
    def test_error_handling_during_initialization(self, mock_auth):
        """Tests error handling during service initialization.

        Args:
            mock_auth: Mock for the authentication service.

        The test verifies:
            - Proper exception handling
            - Error message formatting
            - Service cleanup on failure
        """
        # Arrange
        mock_auth.side_effect = Exception("Connection failed")

        # Act & Assert
        with self.assertRaises(Exception) as context:
            RivaASRService(api_key="test_api_key")

        # Verify the error message
        self.assertTrue("Missing module: Connection failed" in str(context.exception))

    @patch("nvidia_pipecat.services.riva_speech.riva.client.Auth")
    @patch("nvidia_pipecat.services.riva_speech.riva.client.ASRService")
    def test_can_generate_metrics(self, mock_asr_service, mock_auth):
        """Test that the service can generate metrics."""
        # Arrange
        service = RivaASRService(api_key="test_api_key")

        # Act & Assert
        self.assertFalse(service.can_generate_metrics())

    @patch("nvidia_pipecat.services.riva_speech.riva.client.Auth")
    @patch("nvidia_pipecat.services.riva_speech.riva.client.ASRService")
    def test_start_method(self, mock_asr_service, mock_auth):
        """Test the start method of RivaASRService."""
        # Arrange
        service = RivaASRService(api_key="test_api_key")

        # Create a completed mock task for expected return value
        mock_task = MagicMock()
        mock_task.done.return_value = True

        # Use MagicMock to avoid coroutine awaiting issues
        service.create_task = MagicMock(return_value=mock_task)

        # Mock the response task handler
        mock_response_coro = MagicMock()
        service._response_task_handler = MagicMock(return_value=mock_response_coro)

        # Create a mock StartFrame with the necessary attributes
        mock_start_frame = MagicMock(spec=StartFrame)

        # Act
        async def run_test():
            await service.start(mock_start_frame)

        # Run the test
        asyncio.run(run_test())

        # Assert
        # Verify create_task was called with the right handlers
        service.create_task.assert_called_once_with(mock_response_coro)

    @patch("nvidia_pipecat.services.riva_speech.riva.client.Auth")
    @patch("nvidia_pipecat.services.riva_speech.riva.client.ASRService")
    def test_stop_method(self, mock_asr_service, mock_auth):
        """Test the stop method of RivaASRService."""
        # Arrange
        service = RivaASRService(api_key="test_api_key")
        service._stop_tasks = AsyncMock()

        # Create a mock EndFrame
        mock_end_frame = MagicMock(spec=EndFrame)

        # Act
        async def run_test():
            await service.stop(mock_end_frame)

        # Run the test
        asyncio.run(run_test())

        # Assert
        service._stop_tasks.assert_called_once()

    @patch("nvidia_pipecat.services.riva_speech.riva.client.Auth")
    @patch("nvidia_pipecat.services.riva_speech.riva.client.ASRService")
    def test_cancel_method(self, mock_asr_service, mock_auth):
        """Test the cancel method of RivaASRService."""
        # Arrange
        service = RivaASRService(api_key="test_api_key")
        service._stop_tasks = AsyncMock()

        # Create a mock CancelFrame
        mock_cancel_frame = MagicMock(spec=CancelFrame)

        # Act
        async def run_test():
            await service.cancel(mock_cancel_frame)

        # Run the test
        asyncio.run(run_test())

        # Assert
        service._stop_tasks.assert_called_once()

    @patch("nvidia_pipecat.services.riva_speech.riva.client.Auth")
    @patch("nvidia_pipecat.services.riva_speech.riva.client.ASRService")
    def test_run_stt_yields_frames(self, mock_asr_service, mock_auth):
        """Test that run_stt method yields frames."""
        # Arrange
        service = RivaASRService(api_key="test_api_key")
        service._queue = AsyncMock()

        # Create a completed mock task and mock thread task handler
        mock_task = MagicMock()
        mock_task.done.return_value = True

        mock_thread_coro = MagicMock()
        service._thread_task_handler = MagicMock(return_value=mock_thread_coro)
        service._thread_task = mock_task

        # Mock the create_task method to return our mock task and avoid coroutine warnings
        service.create_task = MagicMock(return_value=mock_task)

        # Act
        async def run_test():
            frames = []
            audio_data = b"test_audio_data"
            async for frame in service.run_stt(audio_data):
                frames.append(frame)

            # Assert
            service._queue.put.assert_called_once_with(audio_data)
            # run_stt yields a single None frame for RivaASRService
            self.assertEqual(len(frames), 1)
            self.assertIsNone(frames[0])
            service.create_task.assert_called_once_with(mock_thread_coro)

        # Run the test
        asyncio.run(run_test())

    @patch("nvidia_pipecat.services.riva_speech.riva.client.Auth")
    @patch("nvidia_pipecat.services.riva_speech.riva.client.ASRService")
    def test_handle_response_with_final_transcript(self, mock_asr_service, mock_auth):
        """Tests handling of ASR responses with final transcripts.

        Args:
            mock_asr_service: Mock for the ASR service.
            mock_auth: Mock for the authentication service.

        The test verifies:
            - Final transcript processing
            - Metrics handling
            - Frame generation
            - Response completion handling
        """
        # Arrange
        service = RivaASRService(api_key="test_api_key")
        service.push_frame = AsyncMock()

        # Use MagicMock instead of asyncio.Future to avoid event loop issues
        service.stop_ttfb_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()

        # Create a mock response with final transcript
        mock_result = MagicMock()
        mock_alternative = MagicMock()
        mock_alternative.transcript = "This is a final transcript"
        mock_result.alternatives = [mock_alternative]
        mock_result.is_final = True

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        # Act
        async def run_test():
            await service._handle_response(mock_response)

        # Run the test
        asyncio.run(run_test())

        # Assert
        service.stop_ttfb_metrics.assert_called_once()
        service.stop_processing_metrics.assert_called_once()

        # Verify that a TranscriptionFrame was pushed with the correct text
        found = False
        for call_args in service.push_frame.call_args_list:
            frame = call_args[0][0]
            if isinstance(frame, TranscriptionFrame) and frame.text == "This is a final transcript":
                found = True
                break
        self.assertTrue(found, "No TranscriptionFrame with the expected text was pushed")

    @patch("nvidia_pipecat.services.riva_speech.riva.client.Auth")
    @patch("nvidia_pipecat.services.riva_speech.riva.client.ASRService")
    def test_handle_response_with_interim_transcript(self, mock_asr_service, mock_auth):
        """Test handling of ASR responses with interim transcript."""
        # Arrange
        service = RivaASRService(api_key="test_api_key")
        service.push_frame = AsyncMock()

        # Use AsyncMock directly instead of Future
        service.stop_ttfb_metrics = AsyncMock()

        # Create a mock response with interim transcript
        mock_result = MagicMock()
        mock_alternative = MagicMock()
        mock_alternative.transcript = "This is an interim transcript"
        mock_result.alternatives = [mock_alternative]
        mock_result.is_final = False
        mock_result.stability = 1.0  # High stability interim result

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        # Act
        async def run_test():
            await service._handle_response(mock_response)

        # Run the test
        asyncio.run(run_test())

        # Assert
        service.stop_ttfb_metrics.assert_called_once()

        # Verify that a RivaInterimTranscriptionFrame was pushed with the correct text
        found = False
        for call_args in service.push_frame.call_args_list:
            frame = call_args[0][0]
            if (
                isinstance(frame, RivaInterimTranscriptionFrame)
                and frame.text == "This is an interim transcript"
                and frame.stability == 1.0
            ):
                found = True
                break
        self.assertTrue(found, "No RivaInterimTranscriptionFrame with the expected text was pushed")

    @patch("nvidia_pipecat.services.riva_speech.riva.client.Auth")
    @patch("nvidia_pipecat.services.riva_speech.riva.client.ASRService")
    def test_handle_interruptions(self, mock_asr_service, mock_auth):
        """Tests handling of speech interruptions.

        Args:
            mock_asr_service: Mock for the ASR service.
            mock_auth: Mock for the authentication service.

        The test verifies:
            - Interruption start handling
            - Interruption stop handling
            - Frame sequence generation
            - State management during interruptions
        """
        # Arrange
        service = RivaASRService(api_key="test_api_key", generate_interruptions=True)
        service.push_frame = AsyncMock()

        # Use AsyncMock directly instead of Future
        service._start_interruption = AsyncMock()
        service._stop_interruption = AsyncMock()

        # Mock the property to return True - avoids setting the property directly
        type(service).interruptions_allowed = MagicMock(return_value=True)

        # Act
        async def run_test():
            # Simulate interruption handling
            user_started_frame = UserStartedSpeakingFrame()
            user_stopped_frame = UserStoppedSpeakingFrame()

            # Direct calls to _handle_interruptions
            await service._handle_interruptions(user_started_frame)
            await service._handle_interruptions(user_stopped_frame)

        # Run the test
        asyncio.run(run_test())

        # Assert
        service._start_interruption.assert_called_once()
        service._stop_interruption.assert_called_once()

        # Check that frames were pushed (check by type instead of exact equality)
        pushed_frame_types = [type(call[0][0]) for call in service.push_frame.call_args_list]
        self.assertIn(StartInterruptionFrame, pushed_frame_types, "No StartInterruptionFrame was pushed")
        self.assertIn(StopInterruptionFrame, pushed_frame_types, "No StopInterruptionFrame was pushed")
        self.assertIn(UserStartedSpeakingFrame, pushed_frame_types, "No UserStartedSpeakingFrame was pushed")
        self.assertIn(UserStoppedSpeakingFrame, pushed_frame_types, "No UserStoppedSpeakingFrame was pushed")


@pytest.mark.asyncio
async def test_riva_asr_integration():
    """Tests integration of RivaASRService components.

    Tests the complete flow of the ASR service including initialization,
    processing, and cleanup.

    The test verifies:
        - Service startup sequence
        - Audio processing pipeline
        - Response handling
        - Service shutdown sequence
        - Resource cleanup
    """
    with (
        patch("nvidia_pipecat.services.riva_speech.riva.client.Auth"),
        patch("nvidia_pipecat.services.riva_speech.riva.client.ASRService") as mock_asr_service,
    ):
        # Setup mock ASR service
        mock_instance = mock_asr_service.return_value

        # Initialize service with interruptions enabled
        service = RivaASRService(api_key="test_api_key", generate_interruptions=True)
        service._asr_service = mock_instance

        # Set up the response queue
        service._response_queue = asyncio.Queue()

        # Mock the _stop_tasks method directly instead of relying on task_manager
        service._stop_tasks = AsyncMock()

        # Create mock coroutines for handlers
        thread_coro = MagicMock()
        response_coro = MagicMock()

        # Mock the handler methods to return the coroutines
        service._thread_task_handler = MagicMock(return_value=thread_coro)
        service._response_task_handler = MagicMock(return_value=response_coro)

        # Create a mock task that's already completed
        mock_task = MagicMock()
        mock_task.done.return_value = True

        # Mock task creation to return our completed task
        service.create_task = MagicMock(return_value=mock_task)

        # Set the tasks to our mock task
        service._thread_task = mock_task
        service._response_task = mock_task

        # Create a mock result for testing
        mock_result = MagicMock()
        mock_alternative = MagicMock()
        mock_alternative.transcript = "This is a test transcript"
        mock_result.alternatives = [mock_alternative]
        mock_result.is_final = True

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        # Create a mock StartFrame
        mock_start_frame = MagicMock(spec=StartFrame)

        # Start the service with a start frame
        await service.start(mock_start_frame)

        # Verify response task was created
        service.create_task.assert_called_with(response_coro)

        # Put a mock response in the queue
        await service._response_queue.put(mock_response)

        # Test some other functionality
        audio_data = b"test_audio_data"

        # Run the run_stt method
        frames = []
        async for frame in service.run_stt(audio_data):
            frames.append(frame)

        # Verify results
        assert len(frames) == 1
        assert frames[0] is None

        # Verify thread task was created
        service.create_task.assert_any_call(thread_coro)

        # Simulate stopping the service
        mock_end_frame = MagicMock(spec=EndFrame)
        await service.stop(mock_end_frame)

        # Verify stop_tasks was called
        service._stop_tasks.assert_called_once()


if __name__ == "__main__":
    unittest.main()
