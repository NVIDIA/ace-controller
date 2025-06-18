# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""NVIDIA Riva Neural Machine Translation (NMT) service implementation.

This module provides integration with NVIDIA Riva's NMT service for text translation
between different languages. It supports:
- Real-time text translation
- Multiple language pairs
- Integration with LLM and sentence aggregation pipelines
"""

import re

from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService
from pipecat.transcriptions.language import Language

try:
    import riva.client
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use nvidia rivaskills NMT, you need to `pip install pipecat-ai[riva]`.")
    raise Exception(f"Missing module: {e}") from e


class RivaNMTService(AIService):
    """Base class for services using Riva NMT.

    Handles translation of text between languages using NVIDIA's Riva Neural Machine
    Translation service. Requires Riva NMT models to be deployed following:
    https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide/nmt.html
    """

    def __init__(
        self,
        source_language: Language,
        target_language: Language,
        model_name: str = "",
        server: str = "localhost:50051",
        **kwargs,
    ):
        """Initialize the Riva NMT service.

        Args:
            source_language: Source language for translation.
            target_language: Target language for translation.
            model_name: Name of the RIVA translation model. Empty string will
                auto-select an available model.
            server: Riva server address.
            **kwargs: Additional arguments for AIService parent class.

        Raises:
            Exception: If source_language or target_language is not provided.
        """
        if not source_language:
            raise Exception("No source language provided for the translation..")
        if not target_language:
            raise Exception("No target language provided for the translation..")
        super().__init__(**kwargs)
        self.set_model_name(model_name)
        self.source_language = source_language
        self.target_language = target_language
        self.llm_full_response_started = False
        self.llm_full_response = ""
        self.auth = riva.client.Auth(uri=server)
        self.riva_nmt_client = riva.client.NeuralMachineTranslationClient(self.auth)

    async def translate_text(self, text: str = "") -> tuple[str | None, str | None]:
        """Translates text using Riva NMT service.

        Args:
            text: The text to translate. Must not be empty.

        Returns:
            A tuple containing:
                - str | None: Translated text if successful, None if failed
                - str | None: Error message if failed, None if successful

        Raises:
            Exception: If no input text is provided for translation.
        """
        try:
            if not text:
                raise Exception("No input text provided for the translation..")

            logger.debug(f"Received text: {text}")
            logger.debug(f"Translating the text from {self.source_language} to {self.target_language}")
            response = self.riva_nmt_client.translate(
                [text], self._model_name, self.source_language, self.target_language
            )
            logger.debug(f"Final translated text: {response.translations[0].text}")
            return response.translations[0].text, None
        except Exception as e:
            logger.error(
                f"Error while translating the text from {self.source_language} to {self.target_language}, Error: {e}"
            )
            return (
                None,
                f"Error while translating the text from {self.source_language} to {self.target_language}, Error: {e}",
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Processes incoming frames for translation.

        Handles different frame types:
            - TranscriptionFrame: Translates text and pushes LLMMessagesFrame
            - LLMFullResponseStartFrame: Marks start of LLM response
            - LLMFullResponseEndFrame: Translates accumulated response and pushes TextFrame
            - TextFrame: Accumulates text during LLM response

        Args:
            frame: Frame to process.
            direction: Direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            await self.start_processing_metrics()
            translated_text, err = await self.translate_text(frame.text)
            await self.stop_processing_metrics()
            if err is not None:
                await self.push_error(ErrorFrame(err))
            else:
                messages = [{"role": "system", "content": translated_text}]
                await self.push_frame(LLMMessagesFrame(messages))
        elif isinstance(frame, LLMFullResponseStartFrame):
            self.llm_full_response_started = True
        elif isinstance(frame, LLMFullResponseEndFrame):
            self.llm_full_response_started = False
            # Removing period, question mark, exclamation point, colon, or semicolon
            # as these match end of sentence regex in
            # _process_text_frame() method of TTSService of pipecat/services/ai_services.py
            # and TTS response gets truncated.
            await self.start_processing_metrics()
            self.llm_full_response = re.sub("[.?!:;]", "", self.llm_full_response)
            translated_text, err = await self.translate_text(self.llm_full_response)
            await self.stop_processing_metrics()
            if err is not None:
                await self.push_error(ErrorFrame(err))
            else:
                await self.push_frame(TextFrame(translated_text + "."))
            self.llm_full_response = ""
        elif self.llm_full_response_started and isinstance(frame, TextFrame):
            self.llm_full_response += frame.text
        else:
            await self.push_frame(frame, direction)
