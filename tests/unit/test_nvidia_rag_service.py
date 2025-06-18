# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the NvidiaRAGService processor."""

import httpx
import pytest
from loguru import logger
from pipecat.frames.frames import ErrorFrame, LLMMessagesFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from nvidia_pipecat.frames.nvidia_rag import NvidiaRAGCitation, NvidiaRAGCitationsFrame, NvidiaRAGSettingsFrame
from nvidia_pipecat.services.nvidia_rag import NvidiaRAGService
from tests.unit.utils import FrameStorage, ignore_ids, run_interactive_test, run_test


class MockResponse:
    """Mock response class for testing HTTP responses.

    Attributes:
        json: The JSON response data.
    """

    def __init__(self, json):
        """Initialize MockResponse with JSON data.

        Args:
            json: The JSON response data.
        """
        self.json = json

    async def aclose(self):
        """Mock aclose method."""
        pass

    async def aiter_lines(self):
        """Simulate chunk iteration for streaming response.

        Yields:
            tuple: A tuple containing data and None.
        """
        yield self.json


@pytest.mark.asyncio
async def test_nvidia_rag_service(mocker):
    """Test NvidiaRAGService functionality with various test cases.

    Tests different RAG service behaviors including successful responses,
    citation handling, and error conditions.

    Args:
        mocker: Pytest mocker fixture for mocking HTTP responses.

    The test verifies:
        - Successful responses without citations
        - Successful responses with citations
        - Error handling for empty collection names
        - Error handling for empty queries
        - Error handling for incorrect message roles
    """
    testcases = {
        "Success without citations": {
            "collection_name": "collection123",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful Large Language Model. "
                    "Your goal is to demonstrate your capabilities in a succinct way. "
                    "Your output will be converted to audio so don't include special characters in your answers. "
                    "Respond to what the user said in a creative and helpful way.",
                },
            ],
            "response_json": 'data: {"id":"a886cc44-e2ce-4ea3-95f0-9ffb1171adb1",'
            '"choices":[{"index":0,"message":{"role":"assistant","content":"this is rag response content"},'
            '"delta":{"role":"assistant","content":""},"finish_reason":"[DONE]"}]}',
            "result_frame": TextFrame("this is rag response content"),
        },
        "Success with citations": {
            "collection_name": "collection123",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful Large Language Model. "
                    "Your goal is to demonstrate your capabilities in a succinct way. "
                    "Your output will be converted to audio so don't include special characters in your answers. "
                    "Respond to what the user said in a creative and helpful way.",
                },
            ],
            "response_json": 'data: {"id":"a886cc44-e2ce-4ea3-95f0-9ffb1171adb1",'
            '"choices":[{"index":0,"message":{"role":"assistant","content":"this is rag response content"},'
            '"delta":{"role":"assistant","content":""},"finish_reason":"[DONE]"}], "citations":{"total_results":0,'
            '"results":[{"document_id": "", "content": "this is rag citation content", "document_type": "text",'
            ' "document_name": "", "metadata": "", "score": 0.0}]}}',
            "result_frame": NvidiaRAGCitationsFrame(
                [
                    NvidiaRAGCitation(
                        document_id="",
                        document_type="text",
                        metadata="",
                        score=0.0,
                        document_name="",
                        content=b"this is rag citation content",
                    )
                ]
            ),
        },
        "Fail due to empty collection name": {
            "collection_name": "",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful Large Language Model. "
                    "Your goal is to demonstrate your capabilities in a succinct way. "
                    "Your output will be converted to audio so don't include special characters in your answers. "
                    "Respond to what the user said in a creative and helpful way.",
                },
            ],
            "result_frame": ErrorFrame(
                "An error occurred in http request to RAG endpoint, Error: No query or collection name is provided.."
            ),
        },
        "Fail due to empty query": {
            "collection_name": "collection123",
            "messages": [
                {
                    "role": "system",
                    "content": "",
                },
            ],
            "result_frame": ErrorFrame(
                "An error occurred in http request to RAG endpoint, Error: No query or collection name is provided.."
            ),
        },
        "Fail due to incorrect role": {
            "collection_name": "collection123",
            "messages": [
                {
                    "role": "tool",
                    "content": "",
                },
            ],
            "result_frame": ErrorFrame(
                "An error occurred in http request to RAG endpoint, Error: Unexpected role tool found!"
            ),
        },
    }

    for tc_name, tc_data in testcases.items():
        logger.info(f"Verifying test case: {tc_name}")

        resp = None
        if "response_json" in tc_data:
            resp = MockResponse(tc_data["response_json"])
        else:
            resp = MockResponse("{}")

        mocker.patch("httpx.AsyncClient.post", return_value=resp)

        rag = NvidiaRAGService(collection_name=tc_data["collection_name"])
        storage1 = FrameStorage()
        storage2 = FrameStorage()
        context_aggregator = rag.create_context_aggregator(OpenAILLMContext(tc_data["messages"]))

        pipeline = Pipeline([context_aggregator.user(), storage1, rag, storage2, context_aggregator.assistant()])

        async def test_routine(task: PipelineTask, test_data=tc_data, s1=storage1, s2=storage2):
            await task.queue_frame(LLMMessagesFrame(test_data["messages"]))

            # Wait for the result frame
            if "ErrorFrame" in test_data["result_frame"].name:
                await s1.wait_for_frame(ignore_ids(test_data["result_frame"]))
            else:
                await s2.wait_for_frame(ignore_ids(test_data["result_frame"]))

        await run_interactive_test(pipeline, test_coroutine=test_routine)

        # Verify the frames in storage1
        for frame_history_entry in storage1.history:
            if frame_history_entry.frame.name.startswith("ErrorFrame"):
                assert frame_history_entry.frame == ignore_ids(tc_data["result_frame"])

        # Verify the frames in storage2
        for frame_history_entry in storage2.history:
            if (
                frame_history_entry.frame.name.startswith("TextFrame")
                and tc_data["result_frame"].__str__().startswith("TextFrame")
                or frame_history_entry.frame.name.startswith("NvidiaRAGCitationsFrame")
                and tc_data["result_frame"].__str__().startswith("NvidiaRAGCitationsFrame")
            ):
                assert frame_history_entry.frame == ignore_ids(tc_data["result_frame"])


@pytest.mark.asyncio
async def test_rag_service_sharing_session():
    """Test session sharing behavior between NvidiaRAGService instances.

    Tests the HTTP client session management across multiple RAG service
    instances.

    The test verifies:
        - Session sharing between instances with same parameters
        - Separate session handling for custom sessions
        - Proper session cleanup
    """
    rags = []
    rags.append(NvidiaRAGService(collection_name="collection_1"))
    rags.append(NvidiaRAGService(collection_name="collection_1"))

    initial_session = rags[1].shared_session

    for rag in rags:
        assert rag.shared_session is initial_session

    new_session = httpx.AsyncClient()
    rags.append(NvidiaRAGService(collection_name="collection_1", session=new_session))

    assert rags[0].shared_session is initial_session
    assert rags[1].shared_session is initial_session
    assert rags[2].shared_session is new_session

    await new_session.aclose()
    for r in rags:
        await r.cleanup()


@pytest.mark.asyncio
async def test_nvidia_rag_settings_frame_update(mocker):
    """Tests NvidiaRAGService settings update functionality.

    Tests the processing of NvidiaRAGSettingsFrame for dynamic configuration
    updates.

    Args:
        mocker: Pytest mocker fixture for mocking HTTP responses.

    The test verifies:
        - Collection name updates
        - Server URL updates
        - Settings frame propagation
    """
    mocker.patch("httpx.AsyncClient.post", return_value="")

    rag_settings_frame = NvidiaRAGSettingsFrame(
        settings={"collection_name": "nvidia_blogs", "rag_server_url": "http://10.41.23.247:8081"}
    )
    rag = NvidiaRAGService(collection_name="collection123")

    frames_to_send = [rag_settings_frame]
    expected_down_frames = [rag_settings_frame]

    await run_test(
        rag,
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )

    assert rag.collection_name == "nvidia_blogs"
    assert rag.rag_server_url == "http://10.41.23.247:8081"
