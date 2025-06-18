# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""NVIDIA Retrieval-Augmented Generation (RAG) service implementation.

Integrates with NVIDIA's Retrieval-Augmented Generation service to enhance responses
by incorporating knowledge from external documents. Features include:
    - Document collection management
    - Real-time retrieval and citation
    - OpenAI-compatible LLM interface
    - Configurable retrieval parameters
"""

import json

import httpx
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    StartInterruptionFrame,
    TextFrame,
    VisionImageRawFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService

from nvidia_pipecat.frames.nvidia_rag import NvidiaRAGCitation, NvidiaRAGCitationsFrame, NvidiaRAGSettingsFrame


class NvidiaRAGService(OpenAILLMService):
    """This is the base class for all services that use NVIDIA RAG/GenerativeAIExamples.

    Requires deployed NVIDIA RAG server. For deployment instructions see:
    https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/quickstart.md

    Attributes:
        collection_name: Document collection identifier.
        rag_server_url: RAG API endpoint URL.
        stop_words: Words that stop LLM generation.
        temperature: Controls response randomness (0-1).
        top_p: Token probability threshold (0-1).
        max_tokens: Maximum response length.
        use_knowledge_base: Whether to use RAG retrieval.
        vdb_top_k: Number of chunks to retrieve.
        reranker_top_k: Number of chunks to rerank.
        enable_citations: Whether to return citations.
        suffix_prompt: Text appended to last user message.
    """

    _shared_session: httpx.AsyncClient | None = None

    def __init__(
        self,
        collection_name: str,
        rag_server_url: str = "http://localhost:8081",
        stop_words: list | None = None,
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_tokens: int = 1000,
        use_knowledge_base: bool = True,
        vdb_top_k: int = 20,
        reranker_top_k: int = 4,
        enable_citations: bool = True,
        suffix_prompt: str | None = None,
        session: httpx.AsyncClient | None = None,
        **kwargs,
    ):
        """Initialize the NVIDIA RAG service.

        Args:
            collection_name: Document collection identifier.
            rag_server_url: RAG API endpoint URL.
            stop_words: Words that stop LLM generation.
            temperature: Controls response randomness (0-1).
            top_p: Token probability threshold (0-1).
            max_tokens: Maximum response length.
            use_knowledge_base: Whether to use RAG retrieval.
            vdb_top_k: Number of chunks to retrieve.
            reranker_top_k: Number of chunks to rerank.
            enable_citations: Whether to return citations.
            suffix_prompt: Text appended to last user message.
            session: Optional httpx.AsyncClient. Creates new if None.
            **kwargs: Additional arguments passed to OpenAILLMService.
        """
        super().__init__(api_key="", **kwargs)
        self.collection_name = collection_name
        self.rag_server_url = rag_server_url
        if stop_words is None:
            stop_words = []
        self.stop_words = stop_words
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.use_knowledge_base = use_knowledge_base
        self.vdb_top_k = vdb_top_k
        self.reranker_top_k = reranker_top_k
        self.enable_citations = enable_citations
        self.suffix_prompt = suffix_prompt
        self._external_client_session = None
        self._current_task = None

        if session is not None:
            self._external_client_session = session

    @property
    def shared_session(self) -> httpx.AsyncClient:
        """Get the shared HTTP client session.

        Returns:
            httpx.AsyncClient: The shared session for making HTTP requests.
            Creates a new session if none exists and no external session was provided.
        """
        if self._external_client_session is not None:
            return self._external_client_session

        if NvidiaRAGService._shared_session is None:
            NvidiaRAGService._shared_session = httpx.AsyncClient()
        return NvidiaRAGService._shared_session

    @shared_session.setter
    def shared_session(self, shared_session: httpx.AsyncClient):
        """Set the shared HTTP client session.

        Args:
            shared_session: The httpx.AsyncClient to use for all instances.
        """
        NvidiaRAGService._shared_session = shared_session

    async def stop(self, frame: EndFrame):
        """Stop the NVIDIA RAG service and cleanup resources.

        Args:
            frame: The EndFrame that triggered the stop.
        """
        await super().stop(frame)
        if self._current_task:
            await self.cancel_task(self._current_task)

    async def cancel(self, frame: CancelFrame):
        """Cancel the NVIDIA RAG service and cleanup resources.

        Args:
            frame: The CancelFrame that triggered the cancellation.
        """
        await super().cancel(frame)
        if self._current_task:
            await self.cancel_task(self._current_task)

    async def cleanup(self):
        """Clean up resources used by the RAG service.

        Closes the shared HTTP client session if it exists and performs parent cleanup.
        """
        await super().cleanup()
        await self._close_client_session()

    async def _close_client_session(self):
        """Close the Client Session if it exists."""
        if NvidiaRAGService._shared_session:
            await NvidiaRAGService._shared_session.aclose()
            NvidiaRAGService._shared_session = None

    async def _get_rag_response(self, request_json: dict):
        resp = await self.shared_session.post(f"{self.rag_server_url}/generate", json=request_json)
        return resp

    async def _process_context(self, context: OpenAILLMContext):
        """Processes LLM context through RAG pipeline.

        Args:
            context: Contains conversation history and settings.

        Raises:
            Exception: If invalid message role or empty query.
        """
        try:
            messages: list[ChatCompletionMessageParam] = context.get_messages()
            chat_details = []

            for msg in messages:
                if msg["role"] != "system" and msg["role"] != "user" and msg["role"] != "assistant":
                    raise Exception(f"Unexpected role {msg['role']} found!")
                chat_details.append({"role": msg["role"], "content": msg["content"]})

            if self.suffix_prompt:
                for i in range(len(chat_details) - 1, -1, -1):
                    if chat_details[i]["role"] == "user":
                        chat_details[i]["content"] += f" {self.suffix_prompt}"
                        break

            logger.debug(f"Chat details: {chat_details}")

            if len(chat_details) == 0 or all(msg["content"] == "" for msg in chat_details) or not self.collection_name:
                raise Exception("No query or collection name is provided..")

            """
            Call the RAG chain server and return the streaming response.
            """
            request_json = {
                "messages": chat_details,
                "use_knowledge_base": self.use_knowledge_base,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "vdb_top_k": self.vdb_top_k,
                "reranker_top_k": self.reranker_top_k,
                "collection_name": self.collection_name,
                "stop": self.stop_words,
                "enable_citations": self.enable_citations,
            }

            await self.start_ttfb_metrics()

            full_response = ""
            resp = await self._get_rag_response(request_json)
            try:
                async for chunk in resp.aiter_lines():
                    await self.stop_ttfb_metrics()

                    citations = []
                    try:
                        chunk = chunk.strip("\n")

                        try:
                            if len(chunk) > 6:
                                parsed = json.loads(chunk[6:])
                                message = parsed["choices"][0]["message"]["content"]
                                if "citations" in parsed:
                                    for citation in parsed["citations"]["results"]:
                                        citations.append(
                                            NvidiaRAGCitation(
                                                document_type=str(citation["document_type"]),
                                                document_id=str(citation["document_id"]),
                                                document_name=str(citation["document_name"]),
                                                content=str(citation["content"]).encode(),
                                                metadata=str(citation["metadata"]),
                                                score=float(citation["score"]),
                                            )
                                        )
                            else:
                                logger.warning(f"Received empty RAG response chunk '{chunk}'.")
                                message = ""

                        except Exception as e:
                            logger.debug(f"Parsing RAG response chunk failed. Error: {e}")
                            message = ""
                        if not message and not citations:
                            continue
                        full_response += message
                        if citations:
                            scores = [citation.score for citation in citations]
                            types = [citation.document_type for citation in citations]
                            logger.debug(f"Received total {len(citations)} RAG citations")
                            logger.debug(f"Received RAG citation types: {types}")
                            logger.debug(f"Received RAG citation scores: {scores}")

                            await self.push_frame(NvidiaRAGCitationsFrame(citations=citations))
                        if message:
                            await self.push_frame(TextFrame(message))
                    except Exception as e:
                        await self.push_error(ErrorFrame("Internal error in RAG stream: " + str(e)))
            finally:
                await resp.aclose()

            logger.debug(f"Full RAG response: {full_response}")

        except Exception as e:
            logger.error(f"An error occurred in http request to RAG endpoint, Error:  {e}")
            await self.push_error(ErrorFrame("An error occurred in http request to RAG endpoint, Error: " + str(e)))

    async def _update_settings(self, settings):
        """Updates service settings.

        Args:
            settings: Dictionary of setting name-value pairs.
        """
        for setting, value in settings.items():
            logger.debug(f"Updating {setting} to {value} via NvidiaRAGSettingsFrame")
            match setting:
                case "collection_name":
                    self.collection_name = value
                case "rag_server_url":
                    self.rag_server_url = value
                case "stop_words":
                    self.stop_words = value
                case "temperature":
                    self.temperature = value
                case "top_p":
                    self.top_p = value
                case "max_tokens":
                    self.max_tokens = value
                case "use_knowledge_base":
                    self.use_knowledge_base = value
                case "vdb_top_k":
                    self.vdb_top_k = value
                case "reranker_top_k":
                    self.reranker_top_k = value
                case "enable_citations":
                    self.enable_citations = value
                case _:
                    logger.warning(f"Unknown setting for NvidiaRAG service: {setting}")

    async def _process_context_and_frames(self, context: OpenAILLMContext):
        """Process context and handle start/end frames with metrics."""
        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        await self._process_context(context)
        await self.stop_processing_metrics()
        await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes pipeline frames.

        Handles settings updates and parent frame processing.

        Args:
            frame: Input frame to process.
            direction: Frame processing direction.
        """
        context = None
        if isinstance(frame, NvidiaRAGSettingsFrame):
            await self._update_settings(frame.settings)
        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            context = OpenAILLMContext()
            context.add_image_frame_message(format=frame.format, size=frame.size, image=frame.image, text=frame.text)
        elif isinstance(frame, StartInterruptionFrame):
            if self._current_task is not None:
                await self.cancel_task(self._current_task)
            await self._start_interruption()
            await self.stop_all_metrics()
            await self.push_frame(frame)
        else:
            await super().process_frame(frame, direction)

        if context:
            new_task = self.create_task(self._process_context_and_frames(context))
            if self._current_task is not None:
                await self.cancel_task(self._current_task)
            self._current_task = new_task
            self._current_task.add_done_callback(lambda _: setattr(self, "_current_task", None))
