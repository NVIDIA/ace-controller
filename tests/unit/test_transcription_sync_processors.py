# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for transcript synchronization processors.

This module contains tests that verify the behavior of transcript synchronization processors,
including both user and bot transcript synchronization with different TTS providers.
The tests ensure proper handling of speech events, transcriptions, and TTS frames.
"""

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame
from pipecat.utils.time import time_now_iso8601

from nvidia_pipecat.frames.transcripts import (
    BotUpdatedSpeakingTranscriptFrame,
    UserStoppedSpeakingTranscriptFrame,
    UserUpdatedSpeakingTranscriptFrame,
)
from nvidia_pipecat.processors.transcript_synchronization import (
    BotTranscriptSynchronization,
    UserTranscriptSynchronization,
)
from tests.unit.utils import ignore_ids, run_test


@pytest.mark.asyncio()
async def test_user_transcript_synchronization_processor():
    """Test the UserTranscriptSynchronization processor functionality.

    Tests the complete flow of user speech transcription synchronization,
    including interim and final transcriptions.

    The test verifies:
        - User speech start/stop handling
        - Interim transcription processing
        - Speaking transcript updates
        - Final transcript generation
        - Frame sequence ordering
        - Multiple speech segment handling
    """
    user_id = ""
    interim_transcript_frames = [
        InterimTranscriptionFrame("Hi", user_id, time_now_iso8601()),
        InterimTranscriptionFrame("there!", user_id, time_now_iso8601()),
        InterimTranscriptionFrame("How are", user_id, time_now_iso8601()),
        InterimTranscriptionFrame("you?", user_id, time_now_iso8601()),
    ]
    finale_transcript_frame1 = TranscriptionFrame("Hi there!", user_id, time_now_iso8601())
    finale_transcript_frame2 = TranscriptionFrame("How are you?", user_id, time_now_iso8601())

    frames_to_send = [
        UserStartedSpeakingFrame(),
        interim_transcript_frames[0],
        interim_transcript_frames[1],
        SleepFrame(0.1),
        UserStoppedSpeakingFrame(),
        finale_transcript_frame1,
        SleepFrame(0.1),
        UserStartedSpeakingFrame(),
        interim_transcript_frames[0],
        interim_transcript_frames[1],
        finale_transcript_frame1,
        interim_transcript_frames[2],
        interim_transcript_frames[3],
        finale_transcript_frame2,
        SleepFrame(0.1),
        UserStoppedSpeakingFrame(),
    ]

    expected_down_frames = [
        ignore_ids(UserStartedSpeakingFrame()),
        ignore_ids(UserUpdatedSpeakingTranscriptFrame("user started speaking")),
        ignore_ids(UserUpdatedSpeakingTranscriptFrame("Hi")),
        ignore_ids(UserUpdatedSpeakingTranscriptFrame("there!")),
        ignore_ids(interim_transcript_frames[0]),
        ignore_ids(interim_transcript_frames[1]),
        ignore_ids(UserStoppedSpeakingFrame()),
        ignore_ids(UserStoppedSpeakingTranscriptFrame("Hi there!")),
        ignore_ids(finale_transcript_frame1),
        ignore_ids(UserStartedSpeakingFrame()),
        ignore_ids(UserUpdatedSpeakingTranscriptFrame("user started speaking")),
        ignore_ids(UserUpdatedSpeakingTranscriptFrame("Hi")),
        ignore_ids(UserUpdatedSpeakingTranscriptFrame("there!")),
        ignore_ids(UserUpdatedSpeakingTranscriptFrame("How are")),
        ignore_ids(UserUpdatedSpeakingTranscriptFrame("you?")),
        ignore_ids(interim_transcript_frames[0]),
        ignore_ids(interim_transcript_frames[1]),
        ignore_ids(finale_transcript_frame1),
        ignore_ids(interim_transcript_frames[2]),
        ignore_ids(interim_transcript_frames[3]),
        ignore_ids(finale_transcript_frame2),
        ignore_ids(UserStoppedSpeakingFrame()),
        ignore_ids(UserStoppedSpeakingTranscriptFrame("Hi there! How are you?")),
    ]

    await run_test(
        UserTranscriptSynchronization("user started speaking"),
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )


@pytest.mark.asyncio()
async def test_bot_transcript_synchronization_processor_with_riva_tts():
    """Test the BotTranscriptSynchronization processor with Riva TTS.

    Tests the synchronization of bot transcripts when using Riva TTS,
    including speech events and interruption handling.

    The test verifies:
        - Bot speech start/stop handling
        - TTS text frame processing
        - Speaking transcript updates
        - Interruption handling
        - Frame sequence ordering
        - Multiple sentence handling
    """
    tts_text_frames = [
        TTSTextFrame("Welcome user!"),
        TTSTextFrame("How are you?"),
        TTSTextFrame("Did you have a nice day?"),
    ]

    frames_to_send = [
        TTSStartedFrame(),  # Bot sentence transcript 1
        tts_text_frames[0],
        TTSStoppedFrame(),
        SleepFrame(0.1),
        BotStartedSpeakingFrame(),  # Start playing sentence 1
        TTSStartedFrame(),  # Bot sentence transcript 2
        tts_text_frames[1],
        TTSStoppedFrame(),
        TTSStartedFrame(),  # Bot sentence transcript 3
        tts_text_frames[2],
        TTSStoppedFrame(),
        SleepFrame(0.1),
        BotStoppedSpeakingFrame(),  # End of playing sentence 1
        SleepFrame(0.1),
        BotStartedSpeakingFrame(),  # Start of playing sentence 2
        SleepFrame(0.1),
        BotStoppedSpeakingFrame(),  # End of playing sentence 2
        SleepFrame(0.1),
        StartInterruptionFrame(),  # User interrupts bot before sentence 3 is played
        TTSStartedFrame(),  # Bot sentence 1 again
        tts_text_frames[0],
        TTSStoppedFrame(),
        SleepFrame(0.1),
        BotStartedSpeakingFrame(),  # Start playing sentence 1
        SleepFrame(0.1),
        BotStoppedSpeakingFrame(),  # End of playing sentence 1
    ]

    expected_down_frames = [
        ignore_ids(TTSStartedFrame()),
        ignore_ids(tts_text_frames[0]),
        ignore_ids(TTSStoppedFrame()),
        ignore_ids(BotStartedSpeakingFrame()),
        ignore_ids(BotUpdatedSpeakingTranscriptFrame("Welcome user!")),
        ignore_ids(TTSStartedFrame()),
        ignore_ids(tts_text_frames[1]),
        ignore_ids(TTSStoppedFrame()),
        ignore_ids(TTSStartedFrame()),
        ignore_ids(tts_text_frames[2]),
        ignore_ids(TTSStoppedFrame()),
        ignore_ids(BotStoppedSpeakingFrame()),
        ignore_ids(BotStartedSpeakingFrame()),
        ignore_ids(BotUpdatedSpeakingTranscriptFrame("How are you?")),
        ignore_ids(BotStoppedSpeakingFrame()),
        ignore_ids(StartInterruptionFrame()),
        ignore_ids(TTSStartedFrame()),
        ignore_ids(tts_text_frames[0]),
        ignore_ids(TTSStoppedFrame()),
        ignore_ids(BotStartedSpeakingFrame()),
        ignore_ids(BotUpdatedSpeakingTranscriptFrame("Welcome user!")),
        ignore_ids(BotStoppedSpeakingFrame()),
    ]

    await run_test(
        BotTranscriptSynchronization(),
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )


@pytest.mark.asyncio()
async def test_bot_transcript_synchronization_processor_with_elevenlabs_tts():
    """Test the BotTranscriptSynchronization processor with ElevenLabs TTS.

    Tests the synchronization of bot transcripts when using ElevenLabs TTS,
    including partial text handling and concatenation.

    The test verifies:
        - Bot speech start/stop handling
        - Partial TTS text processing
        - Speaking transcript updates
        - Text concatenation
        - Frame sequence ordering
        - Complete transcript assembly
    """
    tts_text_frames = [
        TTSTextFrame("Welcome"),
        TTSTextFrame("user!"),
        TTSTextFrame("How"),
        TTSTextFrame("are"),
        TTSTextFrame("you?"),
    ]

    frames_to_send = [
        TTSStartedFrame(),
        tts_text_frames[0],
        SleepFrame(0.1),
        BotStartedSpeakingFrame(),
        tts_text_frames[1],
        tts_text_frames[2],
        tts_text_frames[3],
        tts_text_frames[4],
        SleepFrame(0.1),
        TTSStoppedFrame(),
        SleepFrame(0.1),
        BotStoppedSpeakingFrame(),
    ]

    expected_down_frames = [
        ignore_ids(TTSStartedFrame()),
        ignore_ids(tts_text_frames[0]),
        ignore_ids(BotStartedSpeakingFrame()),
        ignore_ids(BotUpdatedSpeakingTranscriptFrame("Welcome")),
        ignore_ids(BotUpdatedSpeakingTranscriptFrame("Welcome user!")),
        ignore_ids(BotUpdatedSpeakingTranscriptFrame("Welcome user! How")),
        ignore_ids(BotUpdatedSpeakingTranscriptFrame("Welcome user! How are")),
        ignore_ids(BotUpdatedSpeakingTranscriptFrame("Welcome user! How are you?")),
        ignore_ids(tts_text_frames[1]),
        ignore_ids(tts_text_frames[2]),
        ignore_ids(tts_text_frames[3]),
        ignore_ids(tts_text_frames[4]),
        ignore_ids(TTSStoppedFrame()),
        ignore_ids(BotStoppedSpeakingFrame()),
    ]

    await run_test(
        BotTranscriptSynchronization(),
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )
