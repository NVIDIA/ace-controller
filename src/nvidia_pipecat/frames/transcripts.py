# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Transcript frames."""

from dataclasses import dataclass

from pipecat.frames.frames import SystemFrame

from nvidia_pipecat.frames.action import ActionFrame


@dataclass
class UserUpdatedSpeakingTranscriptFrame(SystemFrame, ActionFrame):
    """A frame that contains user's partial transcript.

    Args:
        transcript: The user speech transcript
    """

    transcript: str


@dataclass
class UserStoppedSpeakingTranscriptFrame(SystemFrame, ActionFrame):
    """A frame that contains the final user transcript.

    This frame usually comes after UserStoppedSpeakingFrame.

    Args:
        transcript: The user speech transcript
    """

    transcript: str


@dataclass
class BotUpdatedSpeakingTranscriptFrame(SystemFrame, ActionFrame):
    """A frame that contains bot's partial transcript.

    Args:
        transcript: The bot speech transcript.
    """

    transcript: str
