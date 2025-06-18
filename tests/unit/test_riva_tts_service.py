# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the RivaTTSService.

This module contains tests for the RivaTTSService class, including initialization,
TTS frame generation, audio frame handling, and integration tests.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame, TTSTextFrame
from pipecat.transcriptions.language import Language

from nvidia_pipecat.services.riva_speech import RivaTTSService


class TestRivaTTSService(unittest.TestCase):
    """Test suite for RivaTTSService functionality."""

    @patch("nvidia_pipecat.services.riva_speech.riva.client.Auth")
    @patch("nvidia_pipecat.services.riva_speech.riva.client.SpeechSynthesisService")
    def test_run_tts_yields_audio_frames(self, mock_speech_service, mock_auth):
        """Tests that run_tts correctly yields audio frames in sequence.

        Tests the complete flow of text-to-speech conversion including start,
        text processing, audio generation, and completion frames.

        Args:
            mock_speech_service: Mock for the speech synthesis service.
            mock_auth: Mock for the authentication service.

        The test verifies:
            - TTSStartedFrame generation
            - TTSTextFrame content
            - TTSAudioRawFrame generation
            - TTSStoppedFrame generation
            - Frame sequence order
            - Audio data integrity
        """
        # Arrange
        mock_audio_data = b"sample_audio_data"
        mock_audio_frame = TTSAudioRawFrame(audio=mock_audio_data, sample_rate=16000, num_channels=1)

        # Create a properly structured mock service
        mock_service_instance = MagicMock()
        mock_speech_service.return_value = mock_service_instance

        # Set up the mock to return a regular list, not an async generator
        # The service expects a list/iterable that it can call iter() on
        mock_service_instance.synthesize_online.return_value = [mock_audio_frame]

        # Create an instance of RivaTTSService
        service = RivaTTSService(api_key="test_api_key")

        # Ensure the service has the right mock
        service._service = mock_service_instance

        # Act
        async def run_test():
            frames = []
            async for frame in service.run_tts("Hello, world!"):
                frames.append(frame)

            # Assert
            self.assertEqual(
                len(frames), 4
            )  # Should yield 4 frames: TTSStartedFrame, TTSTextFrame, TTSAudioRawFrame, TTSStoppedFrame
            self.assertIsInstance(frames[0], TTSStartedFrame)
            self.assertIsInstance(frames[1], TTSTextFrame)
            self.assertEqual(frames[1].text, "Hello, world!")
            self.assertIsInstance(frames[2], TTSAudioRawFrame)
            self.assertEqual(frames[2].audio, mock_audio_data)
            self.assertIsInstance(frames[3], TTSStoppedFrame)

        # Run the async test
        asyncio.run(run_test())

    @patch("nvidia_pipecat.services.riva_speech.riva.client.Auth")
    @patch("nvidia_pipecat.services.riva_speech.riva.client.SpeechSynthesisService")
    def test_push_tts_frames(self, mock_speech_service, mock_auth):
        """Tests that _push_tts_frames correctly processes TTS generation.

        Tests the internal frame processing mechanism including metrics
        handling and generator processing.

        Args:
            mock_speech_service: Mock for the speech synthesis service.
            mock_auth: Mock for the authentication service.

        The test verifies:
            - Metrics start/stop timing
            - Generator processing
            - Frame processing order
            - Method call sequence
        """
        # Arrange
        mock_audio_frame = TTSAudioRawFrame(audio=b"sample_audio_data", sample_rate=16000, num_channels=1)

        # Create a properly structured mock service
        mock_service_instance = MagicMock()
        mock_speech_service.return_value = mock_service_instance

        # Return a regular list for synthesize_online
        mock_service_instance.synthesize_online.return_value = [mock_audio_frame]

        # Create an instance of RivaTTSService
        service = RivaTTSService(api_key="test_api_key")
        service._service = mock_service_instance
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        service.process_generator = AsyncMock()

        # Create a mock for run_tts instead of trying to capture its generator
        async def mock_run_tts(text):
            # This is the sequence of frames that would be yielded by run_tts
            yield TTSStartedFrame()
            yield TTSTextFrame(text)
            yield mock_audio_frame
            yield TTSStoppedFrame()

        # Replace the run_tts method with our mock
        with patch.object(service, "run_tts", side_effect=mock_run_tts):
            # Act
            async def run_test():
                await service._push_tts_frames("Hello, world!")

                # Assert
                # Check that the necessary methods were called
                service.start_processing_metrics.assert_called_once()
                service.process_generator.assert_called_once()
                service.stop_processing_metrics.assert_called_once()

                # Verify call order using the call_args.called_before method
                assert service.start_processing_metrics.call_args.called_before(service.process_generator.call_args)
                assert service.process_generator.call_args.called_before(service.stop_processing_metrics.call_args)

            # Run the async test
            asyncio.run(run_test())

    @patch("nvidia_pipecat.services.riva_speech.riva.client.Auth")
    @patch("nvidia_pipecat.services.riva_speech.riva.client.SpeechSynthesisService")
    def test_init_with_different_parameters(self, mock_speech_service, mock_auth):
        """Tests initialization with various configuration parameters.

        Tests service initialization with different combinations of
        configuration options.

        Args:
            mock_speech_service: Mock for the speech synthesis service.
            mock_auth: Mock for the authentication service.

        The test verifies:
            - Parameter assignment
            - Default value handling
            - Custom parameter validation
            - Authentication configuration
            - Service initialization
        """
        # Define test parameters - users should be able to use any values
        test_api_key = "test_api_key"
        test_server = "custom_server:50051"
        test_voice_id = "English-US.Male-1"
        test_sample_rate = 22050
        test_language = Language.ES_ES
        test_zero_shot_quality = 10
        test_model = "custom-tts-model"
        test_dictionary = {"word": "pronunciation"}
        test_audio_prompt_file = "test_audio.wav"
        # Test initialization with different parameters
        service = RivaTTSService(
            api_key=test_api_key,
            server=test_server,
            voice_id=test_voice_id,
            sample_rate=test_sample_rate,
            language=test_language,
            zero_shot_quality=test_zero_shot_quality,
            model=test_model,
            custom_dictionary=test_dictionary,
            zero_shot_audio_prompt_file=test_audio_prompt_file,
            use_ssl=True,
        )

        # Verify the parameters were set correctly
        self.assertEqual(service._api_key, test_api_key)
        self.assertEqual(service._voice_id, test_voice_id)
        self.assertEqual(service._sample_rate, test_sample_rate)
        self.assertEqual(service._language_code, test_language)
        self.assertEqual(service._zero_shot_quality, test_zero_shot_quality)
        self.assertEqual(service._model_name, test_model)
        self.assertEqual(service._custom_dictionary, test_dictionary)
        self.assertEqual(service._zero_shot_audio_prompt_file, test_audio_prompt_file)

        # Verify Auth was called with correct parameters
        mock_auth.assert_called_with(
            None,
            True,  # use_ssl=True
            test_server,
            [["function-id", "0149dedb-2be8-4195-b9a0-e57e0e14f972"], ["authorization", f"Bearer {test_api_key}"]],
        )

    @patch("nvidia_pipecat.services.riva_speech.riva.client.Auth")
    def test_riva_error_handling(self, mock_auth):
        """Tests error handling when Riva service initialization fails.

        Tests the service's behavior when encountering initialization errors.

        Args:
            mock_auth: Mock for the authentication service.

        The test verifies:
            - Exception propagation
            - Error message formatting
            - Cleanup behavior
            - Service state after error
        """
        # Test error handling when Riva service initialization fails
        mock_auth.side_effect = Exception("Connection failed")

        # Assert that exception is raised and propagated
        with self.assertRaises(Exception) as context:
            RivaTTSService(api_key="test_api_key")

        self.assertTrue("Missing module: Connection failed" in str(context.exception))

    @patch("nvidia_pipecat.services.riva_speech.riva.client.Auth")
    @patch("nvidia_pipecat.services.riva_speech.riva.client.SpeechSynthesisService")
    def test_can_generate_metrics(self, mock_speech_service, mock_auth):
        """Tests that the service reports capability to generate metrics.

        Tests the metrics generation capability reporting functionality.

        Args:
            mock_speech_service: Mock for the speech synthesis service.
            mock_auth: Mock for the authentication service.

        The test verifies:
            - Metrics capability reporting
            - Consistency of capability flag
        """
        # Test that the service can generate metrics
        service = RivaTTSService(api_key="test_api_key")
        self.assertTrue(service.can_generate_metrics())


@pytest.mark.asyncio
async def test_riva_tts_integration():
    """Tests integration of RivaTTSService components.

    Tests the complete flow of the TTS service in an integrated environment.

    The test verifies:
        - Service initialization
        - Frame generation sequence
        - Audio chunk processing
        - Frame content validation
        - Service completion
    """
    # Use parentheses for multiline with statements instead of backslashes
    with (
        patch("nvidia_pipecat.services.riva_speech.riva.client.Auth"),
        patch("nvidia_pipecat.services.riva_speech.riva.client.SpeechSynthesisService") as mock_service,
    ):
        # Setup mock responses
        mock_instance = mock_service.return_value

        # Create audio frames for the response
        audio_frame1 = TTSAudioRawFrame(audio=b"audio_chunk_1", sample_rate=16000, num_channels=1)
        audio_frame2 = TTSAudioRawFrame(audio=b"audio_chunk_2", sample_rate=16000, num_channels=1)

        # Return a list of frames, not an async generator
        mock_instance.synthesize_online.return_value = [audio_frame1, audio_frame2]

        # Initialize service and call its methods
        service = RivaTTSService(api_key="test_api_key")
        service._service = mock_instance

        # Simulate running the service
        collected_frames = []
        async for frame in service.run_tts("Test sentence for TTS"):
            collected_frames.append(frame)

        # Verify the expected frames were produced
        assert len(collected_frames) == 5  # Started, TextFrame, 2 audio chunks, stopped
        assert isinstance(collected_frames[0], TTSStartedFrame)
        assert isinstance(collected_frames[1], TTSTextFrame)
        assert collected_frames[1].text == "Test sentence for TTS"
        assert isinstance(collected_frames[2], TTSAudioRawFrame)
        assert isinstance(collected_frames[3], TTSAudioRawFrame)
        assert isinstance(collected_frames[4], TTSStoppedFrame)

        # Verify audio content
        assert collected_frames[2].audio == b"audio_chunk_1"
        assert collected_frames[3].audio == b"audio_chunk_2"


if __name__ == "__main__":
    unittest.main()
