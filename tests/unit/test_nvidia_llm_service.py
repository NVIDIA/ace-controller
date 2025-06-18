# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the NvidiaLLMService.

This module contains tests for the NvidiaLLMService class, focusing on core functionalities:
- Think token filtering (including split tag handling)
- Mistral message preprocessing
- Token usage tracking
- LLM responses and function calls
"""

from unittest.mock import DEFAULT, AsyncMock, patch

import pytest
from pipecat.frames.frames import LLMTextFrame
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from nvidia_pipecat.services.nvidia_llm import NvidiaLLMService


# Custom mocks that mimic OpenAI classes without inheriting from them
class MockCompletionUsage:
    """Mock for CompletionUsage that mimics the structure."""

    def __init__(self, prompt_tokens, completion_tokens, total_tokens):
        """Initialize with token usage counts.

        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            total_tokens: Total number of tokens
        """
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class MockChoiceDelta:
    """Mock for ChoiceDelta that mimics the structure."""

    def __init__(self, content=None, tool_calls=None):
        """Initialize with optional content and tool calls.

        Args:
            content: The text content of the delta
            tool_calls: List of tool calls in the delta
        """
        self.content = content
        self.tool_calls = tool_calls
        self.function_call = None
        self.role = None


class MockChoice:
    """Mock for Choice that mimics the structure."""

    def __init__(self, delta, index=0, finish_reason=None):
        """Initialize with delta, index, and finish reason.

        Args:
            delta: The delta containing content or tool calls
            index: The index of this choice
            finish_reason: Reason for finishing generation
        """
        self.delta = delta
        self.index = index
        self.finish_reason = finish_reason


class MockChatCompletionChunk:
    """Mock for ChatCompletionChunk that mimics the structure."""

    def __init__(self, content=None, usage=None, id="mock-id", tool_calls=None):
        """Initialize a mock of ChatCompletionChunk.

        Args:
            content: The text content in the chunk
            usage: Token usage information
            id: Chunk identifier
            tool_calls: Any tool calls in the chunk
        """
        self.id = id
        self.model = "mock-model"
        self.object = "chat.completion.chunk"
        self.created = 1234567890
        self.usage = usage

        if tool_calls:
            self.choices = [MockChoice(MockChoiceDelta(tool_calls=tool_calls))]
        else:
            self.choices = [MockChoice(MockChoiceDelta(content=content))]


class MockToolCall:
    """Mock for ToolCall."""

    def __init__(self, id="tool-id", function=None, index=0, type="function"):
        """Initialize a tool call.

        Args:
            id: Tool call identifier
            function: The function being called
            index: Index of this tool call
            type: Type of tool call
        """
        self.id = id
        self.function = function
        self.index = index
        self.type = type


class MockFunction:
    """Mock for Function in a tool call."""

    def __init__(self, name="", arguments=""):
        """Initialize a function with name and arguments.

        Args:
            name: Name of the function
            arguments: JSON string of function arguments
        """
        self.name = name
        self.arguments = arguments


class MockAsyncStream:
    """Mock implementation of AsyncStream for testing."""

    def __init__(self, chunks):
        """Initialize with a list of chunks to yield.

        Args:
            chunks: List of chunks to return when iterating
        """
        self.chunks = chunks

    def __aiter__(self):
        """Return self as an async iterator."""
        return self

    async def __anext__(self):
        """Return the next chunk or raise StopAsyncIteration."""
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)


@pytest.mark.asyncio
async def test_mistral_message_preprocessing():
    """Test the Mistral message preprocessing functionality."""
    service = NvidiaLLMService(api_key="test_api_key", mistral_model_support=True)

    # Test with alternating roles (already valid)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]
    processed = service._preprocess_messages_for_mistral(messages)
    assert len(processed) == len(messages)  # No changes needed

    # Test with consecutive messages from same role
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "How are you?"},
    ]
    processed = service._preprocess_messages_for_mistral(messages)
    assert len(processed) == 2  # System + combined user
    assert processed[1]["role"] == "user"
    assert processed[1]["content"] == "Hello How are you?"

    # Test with system message at end
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    processed = service._preprocess_messages_for_mistral(messages)
    assert len(processed) == 2  # User + system
    assert processed[0]["role"] == "user"
    assert processed[1]["role"] == "system"


@pytest.mark.asyncio
async def test_filter_think_token_simple():
    """Test the basic think token filtering functionality."""
    service = NvidiaLLMService(api_key="test_api_key", filter_think_tokens=True)
    service._reset_think_filter_state()

    # Test with simple think content followed by real content
    content = "I'm thinking about the answer</think>This is the actual response"
    filtered = service._filter_think_token(content)
    assert filtered == "This is the actual response"
    assert service._seen_end_tag is True

    # Subsequent content should pass through untouched
    more_content = " and some more text"
    filtered = service._filter_think_token(more_content)
    assert filtered == " and some more text"


@pytest.mark.asyncio
async def test_filter_think_token_split():
    """Test the think token filtering with split tags."""
    service = NvidiaLLMService(api_key="test_api_key", filter_think_tokens=True)
    service._reset_think_filter_state()

    # First part with beginning of tag
    content1 = "Let me think about this problem<"
    filtered1 = service._filter_think_token(content1)
    assert filtered1 == ""  # No output yet
    assert service._partial_tag_buffer == "<"  # Partial tag saved

    # Second part with rest of tag and response
    content2 = "/think>Here's the answer"
    filtered2 = service._filter_think_token(content2)
    assert filtered2 == "Here's the answer"  # Output after tag
    assert service._seen_end_tag is True


@pytest.mark.asyncio
async def test_filter_think_token_no_tag():
    """Test what happens when no think tag is found."""
    service = NvidiaLLMService(api_key="test_api_key", filter_think_tokens=True)
    service._reset_think_filter_state()

    # Add some content in multiple chunks
    filtered1 = service._filter_think_token("This is a response")
    filtered2 = service._filter_think_token(" with no think tag")
    # Verify filtering behavior
    assert filtered1 == filtered2 == ""  # No output during filtering
    assert service._thinking_aggregation == "This is a response with no think tag"
    # Test end-of-processing behavior
    service.push_frame = AsyncMock()
    await service.push_frame(LLMTextFrame(service._thinking_aggregation))
    service._reset_think_filter_state()
    # Verify results
    service.push_frame.assert_called_once()
    assert service.push_frame.call_args.args[0].text == "This is a response with no think tag"
    assert service._thinking_aggregation == ""  # State was reset


@pytest.mark.asyncio
async def test_token_usage_tracking():
    """Test the token usage tracking functionality."""
    service = NvidiaLLMService(api_key="test_api_key")
    service._is_processing = True

    # Test initial accumulation of prompt tokens
    tokens1 = LLMTokenUsage(prompt_tokens=10, completion_tokens=0, total_tokens=10)
    await service.start_llm_usage_metrics(tokens1)
    assert service._prompt_tokens == 10
    assert service._has_reported_prompt_tokens is True

    # Test incremental completion tokens
    tokens2 = LLMTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    await service.start_llm_usage_metrics(tokens2)
    assert service._completion_tokens == 5

    # Test more completion tokens
    tokens3 = LLMTokenUsage(prompt_tokens=10, completion_tokens=8, total_tokens=18)
    await service.start_llm_usage_metrics(tokens3)
    assert service._completion_tokens == 8

    # Test reporting duplicate prompt tokens
    tokens4 = LLMTokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    await service.start_llm_usage_metrics(tokens4)
    assert service._prompt_tokens == 10  # Unchanged
    assert service._completion_tokens == 10


@pytest.mark.asyncio
async def test_process_context_with_think_filtering():
    """Test the full processing with think token filtering."""
    with patch.multiple(
        NvidiaLLMService,
        create_client=DEFAULT,
        _stream_chat_completions=DEFAULT,
        start_ttfb_metrics=DEFAULT,
        stop_ttfb_metrics=DEFAULT,
        push_frame=DEFAULT,
    ) as mocks:
        service = NvidiaLLMService(api_key="test_api_key", filter_think_tokens=True)
        mock_push_frame = mocks["push_frame"]

        # Setup mock stream
        chunks = [
            MockChatCompletionChunk(content="Thinking<"),
            MockChatCompletionChunk(content="/think>Real content"),
            MockChatCompletionChunk(content=" continues"),
        ]
        mocks["_stream_chat_completions"].return_value = MockAsyncStream(chunks)

        # Process context
        context = OpenAILLMContext(messages=[{"role": "user", "content": "Test query"}])
        await service._process_context(context)

        # Verify frame content - empty frames during thinking, content after tag
        frames = [call.args[0].text for call in mock_push_frame.call_args_list]
        assert frames == ["", "Real content", " continues"]


@pytest.mark.asyncio
async def test_process_context_with_function_calls():
    """Test handling of function calls from LLM."""
    with (
        patch.object(NvidiaLLMService, "create_client"),
        patch.object(NvidiaLLMService, "_stream_chat_completions") as mock_stream,
        patch.object(NvidiaLLMService, "has_function") as mock_has_function,
        patch.object(NvidiaLLMService, "call_function") as mock_call_function,
    ):
        service = NvidiaLLMService(api_key="test_api_key")

        # Create tool call chunks that come in parts
        tool_call1 = MockToolCall(
            id="call1", function=MockFunction(name="get_weather", arguments='{"location"'), index=0
        )

        tool_call2 = MockToolCall(id="call1", function=MockFunction(name="", arguments=':"New York"}'), index=0)

        # Create chunks with tool calls
        chunk1 = MockChatCompletionChunk(tool_calls=[tool_call1], id="chunk1")
        chunk2 = MockChatCompletionChunk(tool_calls=[tool_call2], id="chunk2")

        mock_stream.return_value = MockAsyncStream([chunk1, chunk2])
        mock_has_function.return_value = True
        mock_call_function.return_value = None

        # Process a context
        context = OpenAILLMContext(messages=[{"role": "user", "content": "What's the weather in New York?"}])
        await service._process_context(context)

        # Verify function was called with combined arguments
        mock_call_function.assert_called_once()
        args = mock_call_function.call_args.kwargs
        assert args["function_name"] == "get_weather"
        assert args["arguments"] == {"location": "New York"}
        assert args["tool_call_id"] == "call1"


@pytest.mark.asyncio
async def test_process_context_with_mistral_preprocessing():
    """Test processing context with Mistral message preprocessing."""
    with (
        patch.object(NvidiaLLMService, "create_client"),
        patch.object(NvidiaLLMService, "_stream_chat_completions") as mock_stream,
    ):
        service = NvidiaLLMService(api_key="test_api_key", mistral_model_support=True)

        # Setup mock stream
        chunks = [MockChatCompletionChunk(content="I am a response")]
        mock_stream.return_value = MockAsyncStream(chunks)

        # Test 1: Combining consecutive user messages
        context = OpenAILLMContext(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "How are you?"},
            ]
        )
        await service._process_context(context)

        # Verify messages were combined
        processed_messages = context.get_messages()
        assert len(processed_messages) == 2  # System + combined user
        assert processed_messages[1]["role"] == "user"
        assert processed_messages[1]["content"] == "Hello How are you?"

        # Verify stream was called (normal processing)
        mock_stream.assert_called_once()

        # Test 2: System message only - should skip processing
        mock_stream.reset_mock()
        system_only_context = OpenAILLMContext(
            messages=[
                {"role": "system", "content": "You are helpful."},
            ]
        )
        await service._process_context(system_only_context)

        # Verify that stream was not called (processing skipped)
        mock_stream.assert_not_called()
