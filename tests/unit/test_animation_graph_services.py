# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the Animation Graph Service."""

import asyncio
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
import yaml
from loguru import logger
from nvidia_ace.animation_pb2 import (
    AnimationData,
    AudioWithTimeCode,
    Float3,
    Float3ArrayWithTimeCode,
    FloatArrayWithTimeCode,
    QuatF,
    QuatFArrayWithTimeCode,
    SkelAnimation,
)
from nvidia_ace.audio_pb2 import AudioHeader
from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    ErrorFrame,
    StartInterruptionFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask

from nvidia_pipecat.frames.action import (
    FinishedGestureBotActionFrame,
    FinishedPostureBotActionFrame,
    StartedGestureBotActionFrame,
    StartedPostureBotActionFrame,
    StartGestureBotActionFrame,
    StartPostureBotActionFrame,
    StopPostureBotActionFrame,
)
from nvidia_pipecat.frames.animation import (
    AnimationDataStreamRawFrame,
    AnimationDataStreamStartedFrame,
    AnimationDataStreamStoppedFrame,
)
from nvidia_pipecat.services.animation_graph_service import (
    AnimationGraphConfiguration,
    AnimationGraphService,
)
from nvidia_pipecat.utils.logging import setup_default_ace_logging
from nvidia_pipecat.utils.message_broker import MessageBrokerConfig
from tests.unit.utils import FrameStorage, ignore, run_interactive_test

AUDIO_SAMPLE_RATE = 16000
AUDIO_BITS_PER_SAMPLE = 16
AUDIO_CHANNEL_COUNT = 1
AUDIO_BUFFER_FOR_ONE_FRAME_SIZE = int(AUDIO_SAMPLE_RATE * AUDIO_BITS_PER_SAMPLE / 8 * AUDIO_CHANNEL_COUNT / 30.0) + int(
    AUDIO_BITS_PER_SAMPLE / 8
)


def generate_audio_header() -> AudioHeader:
    """Generate an audio header."""
    return AudioHeader(
        samples_per_second=AUDIO_SAMPLE_RATE,
        bits_per_sample=AUDIO_BITS_PER_SAMPLE,
        channel_count=AUDIO_CHANNEL_COUNT,
    )


def generate_animation_data(time_codes: list[float], audio_buffer_size: int) -> AnimationData:
    """Generate an animation data stream raw frame.

    Args:
        time_codes: List of time codes in seconds for the animation keyframes.
        audio_buffer_size: Size of the audio buffer in bytes.

    Returns:
        AnimationData: The generated animation data object.
    """
    # Create a simple animation data object with skeletal animation
    animation_data = AnimationData(
        skel_animation=SkelAnimation(
            # Add blend shape weights with time code
            blend_shape_weights=[
                FloatArrayWithTimeCode(
                    time_code=t,  # time in seconds
                    values=[0.1, 0.2, 0.3],  # example blend shape weights
                )
                for t in time_codes
            ],
            # Add joint translations with time code
            translations=[
                Float3ArrayWithTimeCode(
                    time_code=t,
                    values=[
                        Float3(x=0.0, y=0.0, z=0.0),  # example translation for joint 1
                        Float3(x=0.1, y=0.2, z=0.3),  # example translation for joint 2
                    ],
                )
                for t in time_codes
            ],
            # Add joint rotations with time code
            rotations=[
                QuatFArrayWithTimeCode(
                    time_code=t,
                    values=[
                        QuatF(real=1.0, i=0.0, j=0.0, k=0.0),  # example rotation for joint 1 (identity quaternion)
                        QuatF(
                            real=0.707, i=0.0, j=0.707, k=0.0
                        ),  # example rotation for joint 2 (45° rotation around Y)
                    ],
                )
                for t in time_codes
            ],
            # Add joint scales with time code
            scales=[
                Float3ArrayWithTimeCode(
                    time_code=t,
                    values=[
                        Float3(x=1.0, y=1.0, z=1.0),  # example scale for joint 1 (no scaling)
                        Float3(x=1.1, y=1.1, z=1.1),  # example scale for joint 2 (uniform scaling)
                    ],
                )
                for t in time_codes
            ],
        ),
        # Audio component
        audio=AudioWithTimeCode(
            time_code=0.0,  # time in seconds relative to start_time_code_since_epoch
            audio_buffer=b"\xff" * audio_buffer_size,  # In a real scenario, this would be PCM audio data as bytes
        ),
    )
    return animation_data


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML configuration from a file.

    Args:
        path: Path to the YAML file.

    Returns:
        dict: Parsed YAML content.

    Raises:
        FileNotFoundError: If the YAML file is not found.
    """
    try:
        return yaml.safe_load(path.read_text())

    except FileNotFoundError as error:
        message = "Error: yml config file not found."
        logger.exception(message)
        raise FileNotFoundError(error, message) from error


def read_action_service_config(config_path: Path) -> AnimationGraphConfiguration:
    """Read and parse the animation graph service configuration.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        AnimationGraphConfiguration: Parsed configuration object.
    """
    return AnimationGraphConfiguration(**load_yaml(config_path))


looping_animations_to_test = [
    ("listening", "Listening"),
    ("thinking about something", "Thinking"),
    ("dancing", "Listening"),
]


class MockResponse:
    """Mock HTTP response for testing animation graph service REST endpoints."""

    def __init__(self):
        """Initialize mock response with default status and headers."""
        self.status = HTTPStatus.OK
        self.headers = {"Content-Type": "application/json"}

    async def json(self):
        """Simulate async JSON response.

        Returns:
            dict: Mock response data.
        """
        await asyncio.sleep(0.1)
        return {"response": "OK"}


@pytest.fixture
def anim_graph():
    """Create and configure an Animation Graph Service instance for testing.

    Returns:
        AnimationGraphService: Configured service instance with test configuration.
    """
    animation_config = read_action_service_config(Path("./tests/unit/configs/animation_config.yaml"))
    message_broker_config = MessageBrokerConfig("local_queue", "")

    logger.info("Starting animation graph service initialization...")
    start_time = time.time()
    AnimationGraphService.pregenerate_animation_databases(animation_config)
    init_time = time.time() - start_time
    logger.info(f"Animation graph service initialized in {init_time * 1000:.2f}ms")

    ag = AnimationGraphService(
        animation_graph_rest_url="http://127.0.0.1:8020",
        animation_graph_grpc_target="127.0.0.1:51000",
        message_broker_config=message_broker_config,
        config=animation_config,
    )

    return ag


@pytest.mark.asyncio
@pytest.mark.parametrize("posture", looping_animations_to_test)
@patch("aiohttp.ClientSession.request")
async def test_simple_posture_pipeline(mock_get, posture, anim_graph):
    """Test basic posture pipeline functionality.

    Verifies that the pipeline correctly processes posture commands and
    generates appropriate animation frames.

    Args:
        mock_get: Mock for HTTP client requests.
        posture: Tuple of (natural language description, clip ID) for testing.
        anim_graph: Animation graph service fixture.
    """
    posture_nld, posture_clip_id = posture
    stream_id = "1235"

    # Mocking response from aiohttp.ClientSession.request
    mock_get.return_value.__aenter__.return_value = MockResponse()

    after_storage = FrameStorage()
    before_storage = FrameStorage()
    pipeline = Pipeline([before_storage, anim_graph, after_storage])

    async def test_routine(task: PipelineTask):
        # send events
        await task.queue_frame(TextFrame("Hello"))
        await task.queue_frame(StartPostureBotActionFrame(posture=posture_nld, action_id="posture_1"))

        # wait for the action to start
        started_posture_frame = ignore(StartedPostureBotActionFrame(action_id="posture_1"), "ids", "timestamps")
        await after_storage.wait_for_frame(started_posture_frame)

        # Ensure API endpoints was called in the right way
        mock_get.assert_called_once_with(
            "put",
            f"http://127.0.0.1:8020/streams/{stream_id}/animation_graphs/avatar/variables/posture_state/{posture_clip_id}",
            data="{}",
            headers={"Content-Type": "application/json", "x-stream-id": stream_id},
            params={},
        )
        # ensure we got the text frame as well as a presence started event (and no finished)
        assert after_storage.history[1].frame == ignore(TextFrame("Hello"), "all_ids", "timestamps")
        assert after_storage.history[3].frame == started_posture_frame
        assert len(after_storage.frames_of_type(FinishedPostureBotActionFrame)) == 0

        # stop the action and wait for it to finish
        finished_posture_frame = FinishedPostureBotActionFrame(
            action_id="posture_1", is_success=True, was_stopped=True, failure_reason=""
        )
        await task.queue_frame(StopPostureBotActionFrame(action_id="posture_1"))
        await after_storage.wait_for_frame(ignore(finished_posture_frame, "ids", "timestamps"))
        await before_storage.wait_for_frame(ignore(finished_posture_frame, "ids", "timestamps"))

        # make sure observed frames before and after the processor match
        # (this should be true for all action frame processors)
        assert len(before_storage.history) == len(after_storage.history)
        for before, after in zip(before_storage.history, after_storage.history, strict=False):
            assert before.frame == after.frame

    await run_interactive_test(pipeline, test_coroutine=test_routine, start_metadata={"stream_id": stream_id})


@pytest.mark.asyncio
@patch("aiohttp.ClientSession.request")
async def test_finite_animation(mock_get, anim_graph):
    """Test finite animation execution and completion.

    Verifies that a finite animation:
    - Starts correctly
    - Runs for the expected duration
    - Completes with proper frame sequence
    """
    # Mocking response from aiohttp.ClientSession.request
    mock_get.return_value.__aenter__.return_value = MockResponse()

    storage = FrameStorage()
    pipeline = Pipeline([anim_graph, storage])

    async def test_routine(task: PipelineTask):
        # send events
        await task.queue_frame(StartGestureBotActionFrame(gesture="Test", action_id="g1"))

        # wait for the action to start
        started_gesture_frame = StartedGestureBotActionFrame(action_id="g1")
        await storage.wait_for_frame(ignore(started_gesture_frame, "ids", "timestamps"))
        assert len(storage.frames_of_type(FinishedGestureBotActionFrame)) == 0

        # Action should not be finished
        await asyncio.sleep(0.3)
        assert len(storage.frames_of_type(FinishedGestureBotActionFrame)) == 0

        # Action should now be done
        await asyncio.sleep(0.3)
        assert len(storage.frames_of_type(FinishedGestureBotActionFrame)) == 1

    await run_interactive_test(pipeline, test_coroutine=test_routine, start_metadata={"stream_id": "1235"})


@pytest.mark.asyncio
@patch("aiohttp.ClientSession.request")
async def test_consecutive_postures(mock_get, anim_graph):
    """Test handling of consecutive posture commands.

    Verifies that the service correctly:
    - Processes multiple posture commands in sequence
    - Transitions between postures smoothly
    - Maintains correct frame order and state
    """
    # Mocking response from aiohttp.ClientSession.request
    mock_get.return_value.__aenter__.return_value = MockResponse()

    stream_id = "1235"
    setup_default_ace_logging(stream_id=stream_id, level="TRACE")

    after_storage = FrameStorage()
    before_storage = FrameStorage()
    pipeline = Pipeline([before_storage, anim_graph, after_storage])

    async def test_routine(task: PipelineTask):
        # start first posture
        await task.queue_frame(StartPostureBotActionFrame(posture="talking", action_id="posture_1"))

        # wait for first posture to start
        started_posture_1_frame = ignore(StartedPostureBotActionFrame(action_id="posture_1"), "ids", "timestamps")
        assert started_posture_1_frame.action_id == "posture_1"
        await after_storage.wait_for_frame(started_posture_1_frame)

        # start second posture
        await task.queue_frame(StartPostureBotActionFrame(posture="listening", action_id="posture_2"))

        # wait for second posture to start
        started_posture_2_frame = ignore(StartedPostureBotActionFrame(action_id="posture_2"), "ids", "timestamps")
        assert started_posture_2_frame.action_id == "posture_2"
        await after_storage.wait_for_frame(started_posture_2_frame)

        # Ensure API endpoints was called twice
        assert mock_get.call_count == 2

        # ensure we got the text frame as well as a presence started event (and no finished)
        assert after_storage.history[2].frame == started_posture_1_frame
        assert after_storage.history[4].frame == ignore(
            FinishedPostureBotActionFrame(
                action_id="posture_1", is_success=False, was_stopped=False, failure_reason="Action replaced."
            ),
            "ids",
            "timestamps",
        )
        assert after_storage.history[5].frame == started_posture_2_frame
        assert len(after_storage.frames_of_type(FinishedPostureBotActionFrame)) == 1

        # make sure observed frames before and after the processor match
        assert len(before_storage.history) == len(after_storage.history)
        for before, after in zip(before_storage.history, after_storage.history, strict=False):
            assert before.frame == after.frame

    await run_interactive_test(pipeline, test_coroutine=test_routine, start_metadata={"stream_id": stream_id})


@pytest.mark.asyncio
@patch("aiohttp.ClientSession.request")
async def test_immediate_stop_posture(mock_get, anim_graph):
    """Test immediate posture cancellation.

    Verifies that the service correctly handles immediate posture cancellation:
    - Processes stop command immediately after start
    - Generates appropriate failure frames
    - Maintains correct frame sequence
    """
    # Mocking response from aiohttp.ClientSession.request
    mock_get.return_value.__aenter__.return_value = MockResponse()

    stream_id = "1235"
    setup_default_ace_logging(stream_id=stream_id, level="TRACE")

    after_storage = FrameStorage()
    before_storage = FrameStorage()
    pipeline = Pipeline([before_storage, anim_graph, after_storage])

    async def test_routine(task: PipelineTask):
        # start/stop first posture
        start_posture_1_frame = StartPostureBotActionFrame(posture="talking", action_id="posture_1")
        stop_posture_1_frame = StopPostureBotActionFrame(action_id="posture_1")
        await task.queue_frames([start_posture_1_frame, stop_posture_1_frame])

        # wait for first posture to finish
        finished_posture_1_frame = ignore(
            FinishedPostureBotActionFrame(
                action_id="posture_1", was_stopped=True, is_success=False, failure_reason=ANY
            ),
            "ids",
            "timestamps",
        )
        await after_storage.wait_for_frame(finished_posture_1_frame)

        # check for the correct frame sequence
        assert after_storage.history[-2].frame == ignore(stop_posture_1_frame, "ids", "timestamps")
        assert after_storage.history[-1].frame == finished_posture_1_frame

        # make sure observed frames before and after the processor match
        assert len(before_storage.history) == len(after_storage.history)
        for before, after in zip(before_storage.history, after_storage.history, strict=True):
            assert before.frame == after.frame

    await run_interactive_test(pipeline, test_coroutine=test_routine, start_metadata={"stream_id": stream_id})


@pytest.mark.asyncio
@patch("aiohttp.ClientSession.request")
async def test_stacking_postures_with_interruptions(mock_get, anim_graph):
    """Test stacking postures with interruptions.

    In this test we reproduce a bug that was observed where the override statemachine
    would sometimes fail if we start many postures at the same time and also
    have interruptions.
    """
    # Mocking response from aiohttp.ClientSession.request
    mock_get.return_value.__aenter__.return_value = MockResponse()

    stream_id = "1235"
    setup_default_ace_logging(stream_id=stream_id, level="TRACE")

    after_storage = FrameStorage()
    before_storage = FrameStorage()
    pipeline = Pipeline([before_storage, anim_graph, after_storage])

    async def test_routine(task: PipelineTask):
        N = 200
        for i in range(N):
            await task.queue_frame(StartPostureBotActionFrame(posture="talking", action_id=f"posture_{i}"))
            await asyncio.sleep(0.05)
            if i == 5:
                await task.queue_frame(StartInterruptionFrame())

        final_posture_started = ignore(StartedPostureBotActionFrame(action_id=f"posture_{N - 1}"), "ids", "timestamps")
        await after_storage.wait_for_frame(final_posture_started)

        assert len(after_storage.frames_of_type(FinishedPostureBotActionFrame)) == N - 1

    await run_interactive_test(pipeline, test_coroutine=test_routine, start_metadata={"stream_id": stream_id})


@patch("aiohttp.ClientSession.request")
async def test_handling_animation_data(mock_get, anim_graph):
    """Test animation data stream processing.

    Verifies that the service correctly:
    - Handles animation data stream frames
    - Processes animation data in correct sequence
    - Maintains proper timing between frames
    """
    # Mocking response from aiohttp.ClientSession.request
    mock_get.return_value.__aenter__.return_value = MockResponse()

    # Mocking gRPC stream
    mock_stub = MagicMock()
    stream_mock = AsyncMock()
    stream_mock.write.return_value = "OK"
    stream_mock.done = MagicMock(return_value=False)  #
    mock_stub.PushAnimationDataStream.return_value = stream_mock

    stream_id = "1235"
    anim_graph.stub = mock_stub
    storage = FrameStorage()
    pipeline = Pipeline([anim_graph, storage])

    async def test_routine(task: PipelineTask):
        # send events
        await task.queue_frame(
            AnimationDataStreamStartedFrame(
                audio_header=generate_audio_header(), animation_header=None, action_id="a1", animation_source_id="test"
            )
        )

        for _ in range(15):
            await task.queue_frame(
                AnimationDataStreamRawFrame(
                    animation_data=generate_animation_data([0.0], AUDIO_BUFFER_FOR_ONE_FRAME_SIZE), action_id="a1"
                )
            )
            await asyncio.sleep(1.0 / 35.0)

        await task.queue_frame(AnimationDataStreamStoppedFrame(action_id="a1"))
        assert len(storage.frames_of_type(BotStartedSpeakingFrame)) == 1
        assert len(storage.frames_of_type(BotSpeakingFrame)) >= 1

    await run_interactive_test(pipeline, test_coroutine=test_routine, start_metadata={"stream_id": stream_id})


@patch("aiohttp.ClientSession.request")
async def test_handling_bursted_animation_data(mock_get, anim_graph):
    """Test handling of bursted animation data.

    Verifies that the service correctly handles bursted animation data.
    """
    # Mocking response from aiohttp.ClientSession.request
    mock_get.return_value.__aenter__.return_value = MockResponse()

    # Mocking gRPC stream
    mock_stub = MagicMock()
    stream_mock = AsyncMock()
    stream_mock.write.return_value = "OK"
    stream_mock.done = MagicMock(return_value=False)  #
    mock_stub.PushAnimationDataStream.return_value = stream_mock

    stream_id = "1235"
    anim_graph.stub = mock_stub
    storage = FrameStorage()
    pipeline = Pipeline([anim_graph, storage])

    async def test_routine(task: PipelineTask):
        # send events
        await task.queue_frame(
            AnimationDataStreamStartedFrame(
                audio_header=generate_audio_header(), animation_header=None, action_id="a1", animation_source_id="test"
            )
        )

        for _ in range(30 * 3):
            await task.queue_frame(
                AnimationDataStreamRawFrame(
                    animation_data=generate_animation_data([0.0], AUDIO_BUFFER_FOR_ONE_FRAME_SIZE), action_id="a1"
                )
            )

        await asyncio.sleep(2.5)
        assert stream_mock.done_writing.call_count == 0

        await task.queue_frame(AnimationDataStreamStoppedFrame(action_id="a1"))
        assert len(storage.frames_of_type(BotStartedSpeakingFrame)) == 1
        assert len(storage.frames_of_type(BotSpeakingFrame)) >= 1

    await run_interactive_test(pipeline, test_coroutine=test_routine, start_metadata={"stream_id": stream_id})


@patch("aiohttp.ClientSession.request")
async def test_handling_interrupted_animation_data_stream(mock_get, anim_graph):
    """Test behavior if animation data stream is interrupted."""
    # Mocking response from aiohttp.ClientSession.request
    mock_get.return_value.__aenter__.return_value = MockResponse()

    # Mocking gRPC stream
    mock_stub = MagicMock()
    stream_mock = AsyncMock()
    stream_mock.write.return_value = "OK"
    stream_mock.done = MagicMock(return_value=False)  #
    mock_stub.PushAnimationDataStream.return_value = stream_mock

    stream_id = "1235"
    anim_graph.stub = mock_stub
    storage = FrameStorage()
    pipeline = Pipeline([anim_graph, storage])

    async def test_routine(task: PipelineTask):
        # send events
        await task.queue_frame(
            AnimationDataStreamStartedFrame(
                audio_header=generate_audio_header(), animation_header=None, action_id="a1", animation_source_id="test"
            )
        )

        for _ in range(5):
            await task.queue_frame(
                AnimationDataStreamRawFrame(
                    animation_data=generate_animation_data([0.0], AUDIO_BUFFER_FOR_ONE_FRAME_SIZE), action_id="a1"
                )
            )
            await asyncio.sleep(1.0 / 30.0)

        await asyncio.sleep(2.0)

        assert stream_mock.done_writing.call_count == 1

    await run_interactive_test(pipeline, test_coroutine=test_routine, start_metadata={"stream_id": stream_id})


@patch("aiohttp.ClientSession.request")
async def test_handling_low_fps_animation_data(mock_get, anim_graph):
    """Test animation data stream processing with low FPS.

    Verifies that the service correctly sends out an
    ErrorFrame when the animation data received is below 30 FPS.
    """
    # Mocking response from aiohttp.ClientSession.request
    mock_get.return_value.__aenter__.return_value = MockResponse()

    # Mocking gRPC stream
    mock_stub = MagicMock()
    stream_mock = AsyncMock()
    stream_mock.write.return_value = "OK"
    stream_mock.done = MagicMock(return_value=False)  #
    mock_stub.PushAnimationDataStream.return_value = stream_mock

    stream_id = "1235"
    anim_graph.stub = mock_stub
    storage = FrameStorage()
    pipeline = Pipeline([anim_graph, storage])

    async def test_routine(task: PipelineTask):
        # send events
        await task.queue_frame(
            AnimationDataStreamStartedFrame(
                audio_header=generate_audio_header(), animation_header=None, action_id="a1", animation_source_id="test"
            )
        )

        for _ in range(20):
            await task.queue_frame(
                AnimationDataStreamRawFrame(
                    animation_data=generate_animation_data([0.0], AUDIO_BUFFER_FOR_ONE_FRAME_SIZE), action_id="a1"
                )
            )
            await asyncio.sleep(1.0 / 24.5)

        await task.queue_frame(AnimationDataStreamStoppedFrame(action_id="a1"))

        await storage.wait_for_frame(
            ignore(
                ErrorFrame(error="AnimGraph: Received data stream is behind by more than 0.1s", fatal=False),
                "ids",
                "error",
            )
        )

    await run_interactive_test(pipeline, test_coroutine=test_routine, start_metadata={"stream_id": stream_id})


@pytest.mark.asyncio
async def test_animation_database_search(anim_graph):
    """Test animation database semantic search functionality.

    Verifies that searching for 'wave to the user' correctly returns the 'Goodbye'
    animation clip with appropriate semantic match scores.
    """
    # Get the gesture animation database
    gesture_db = AnimationGraphService.animation_databases["gesture"]

    # Search for the animation
    match = gesture_db.query_one("wave goodbye to the user somehow")

    # Verify we got a match and it's the Goodbye animation
    assert match.animation.id == "Goodbye", "Expected 'wave to the user' to map to 'Goodbye' animation"
    assert match.description_score > 0.5 or match.meaning_score > 0.5, "Expected high semantic match score"
