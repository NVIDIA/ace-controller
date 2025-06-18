"""Unit tests for utility processors."""

import pytest
from pipecat.frames.frames import Frame, TextFrame

from nvidia_pipecat.processors.utils import FrameBlockingProcessor
from tests.unit.utils import ignore_ids, run_test


class TestFrame(Frame):
    """Test frame for testing FrameBlockingProcessor."""

    pass


class TestResetFrame(Frame):
    """Test frame for testing FrameBlockingProcessor reset."""

    pass


@pytest.mark.asyncio()
async def test_frame_blocking_processor():
    """Test that FrameBlockingProcessor blocks frames after threshold.

    Verifies that:
        - Frames are passed through until threshold is reached
        - Frames are blocked after threshold
        - Non-matching frame types are always passed through
    """
    # Create processor that blocks after 2 TextFrames
    processor = FrameBlockingProcessor(block_after_frame=2, frame_type=TextFrame)

    # Create test frames - mix of TextFrames and StartFrames
    frames_to_send = [
        TestFrame(),  # Should pass through
        TextFrame("First"),  # Should pass through
        TextFrame("Second"),  # Should pass through
        TextFrame("Third"),  # Should be blocked
        TestFrame(),  # Should pass through
        TextFrame("Fourth"),  # Should be blocked
    ]

    # Expected frames - all StartFrames and first 2 TextFrames
    expected_down_frames = [
        ignore_ids(TestFrame()),
        ignore_ids(TextFrame("First")),
        ignore_ids(TextFrame("Second")),
        ignore_ids(TestFrame()),
    ]

    await run_test(
        processor,
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )


@pytest.mark.asyncio()
async def test_frame_blocking_processor_with_reset():
    """Test that FrameBlockingProcessor resets counter on reset frame type.

    Verifies that:
        - Counter resets when reset frame type is received
        - Frames are blocked after threshold
        - Counter can be reset multiple times
    """
    # Create processor that blocks after 2 TextFrames and resets on StartFrame
    processor = FrameBlockingProcessor(block_after_frame=2, frame_type=TextFrame, reset_frame_type=TestResetFrame)

    # Create test frames - mix of TextFrames and TestResetFrame
    frames_to_send = [
        TextFrame("First"),  # Should pass through
        TextFrame("Second"),  # Should pass through
        TextFrame("Third"),  # Should be blocked
        TestResetFrame(),  # Should reset counter and pass through
        TextFrame("Fourth"),  # Should pass through (counter reset)
        TextFrame("Fifth"),  # Should pass through
        TextFrame("Sixth"),  # Should be blocked
    ]

    # Expected frames - all TestResetFrame and TextFrames except blocked ones
    expected_down_frames = [
        ignore_ids(TextFrame("First")),
        ignore_ids(TextFrame("Second")),
        ignore_ids(TestResetFrame()),
        ignore_ids(TextFrame("Fourth")),
        ignore_ids(TextFrame("Fifth")),
    ]

    await run_test(
        processor,
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )
