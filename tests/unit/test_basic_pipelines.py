# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Basic functionality tests for pipecat pipelines."""

import asyncio

import pytest
from loguru import logger
from pipecat.frames.frames import StartFrame, TextFrame
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.tests.utils import SleepFrame

from nvidia_pipecat.utils.logging import logger_context, setup_default_ace_logging
from tests.unit.utils import ignore_ids, run_test


@pytest.mark.asyncio()
async def test_simple_pipeline():
    """Example test for testing a pipecat pipeline. This test makes sure the basic pipeline related classes work."""
    aggregator = SentenceAggregator()
    frames_to_send = [TextFrame("Hello, "), TextFrame("world.")]
    expected_down_frames = [ignore_ids(TextFrame("Hello, world."))]

    await run_test(
        aggregator,
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
        start_metadata={"stream_id": "1235"},
    )


@pytest.mark.asyncio()
async def test_pipeline_with_stream_id():
    """Test pipeline creation with a stream_id.

    Verifies that a pipeline can be created with a specific stream_id and that
    the metadata is properly propagated with the StartFrame.
    """
    aggregator = SentenceAggregator()
    frames_to_send = [TextFrame("Hello, "), TextFrame("world.")]
    start_metadata = {"stream_id": "1234"}

    expected_start_frame = ignore_ids(StartFrame())
    expected_start_frame.metadata = start_metadata
    expected_down_frames = [expected_start_frame, ignore_ids(TextFrame("Hello, world."))]

    await run_test(
        aggregator,
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
        start_metadata=start_metadata,
        ignore_start=False,
    )


@pytest.mark.asyncio()
async def test_ace_logger_with_stream_id(capsys):
    """Test ACE logger behavior when stream_id is provided.

    Verifies that the logger correctly handles and displays stream_id in the logs.
    """
    setup_default_ace_logging(level="DEBUG")
    with logger.contextualize(stream_id="1237"):
        aggregator = SentenceAggregator()
        frames_to_send = [TextFrame("Hello, "), TextFrame("world.")]
        expected_down_frames = [ignore_ids(TextFrame("Hello, world."))]
        await run_test(aggregator, frames_to_send=frames_to_send, expected_down_frames=expected_down_frames)

        captured = capsys.readouterr()
        assert "streamId=1237" in captured.err


async def run_pipeline_task(stream: str):
    """Run a test pipeline task with a specific stream identifier.

    Creates and runs a pipeline that processes a sequence of text frames with
    sleep intervals, aggregating them into a single sentence.

    Args:
        stream: The stream identifier to use for the pipeline task.
    """
    REPETITIONS = 5
    frames_to_send = []
    aggregated_str = ""
    for i in range(REPETITIONS):
        frames_to_send.append(TextFrame(f"S{stream}-T{i}"))
        aggregated_str += f"S{stream}-T{i}"
        frames_to_send.append(SleepFrame(0.1))

    frames_to_send.append(TextFrame("."))
    expected_down_frames = [ignore_ids(TextFrame(f"{aggregated_str}."))]
    aggregator = SentenceAggregator()
    await run_test(aggregator, frames_to_send=frames_to_send, expected_down_frames=expected_down_frames)


@pytest.mark.asyncio()
async def test_logging_with_multiple_pipelines_in_same_process(capsys):
    """Test logging behavior with multiple concurrent pipeline streams.

    Verifies that when multiple pipeline tasks are running concurrently in the same
    process, each stream's logs are correctly tagged with its respective stream_id.
    The test ensures proper isolation and identification of log messages across
    different pipeline streams.

    Args:
        capsys: Pytest fixture for capturing system output.
    """
    setup_default_ace_logging(level="TRACE")

    streams = ["777", "abc123"]
    tasks = []

    for stream in streams:
        task = asyncio.create_task(logger_context(run_pipeline_task(stream=stream), stream_id=stream))
        tasks.append(task)

    await asyncio.gather(*tasks)

    # Make sure the correct stream ID is logged for the different coroutines
    captured = capsys.readouterr()
    lines = captured.err.split("\n")
    for line in lines:
        for stream in streams:
            if f"S{stream}" in line:
                assert f"streamId={stream}" in line

    for task in asyncio.all_tasks():
        if "task_handler" in task.get_coro().__name__:
            task.cancel()
