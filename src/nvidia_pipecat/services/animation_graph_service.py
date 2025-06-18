# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Animation graph service for managing avatar animations and interactions with the animation graph service.

You can find more information about the animation graph service here:
https://docs.nvidia.com/ace/animation-graph-microservice/latest/index.html

"""

import asyncio
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from itertools import chain
from pathlib import Path
from typing import Any

import grpc
import torch
from grpc.aio import StreamUnaryCall
from loguru import logger
from nvidia_ace.audio_pb2 import AudioHeader
from nvidia_ace.status_pb2 import Status
from nvidia_animation_graph.animgraph_pb2_grpc import AnimationDataServiceStub
from nvidia_animation_graph.messages_pb2 import AnimationDataStream, AnimationDataStreamHeader, AnimationIds
from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util  # type: ignore

from nvidia_pipecat.frames.action import (
    ActionFrame,
    FinishedFacialGestureBotActionFrame,
    FinishedGestureBotActionFrame,
    FinishedMotionEffectCameraActionFrame,
    FinishedPositionBotActionFrame,
    FinishedPostureBotActionFrame,
    FinishedShotCameraActionFrame,
    StartedFacialGestureBotActionFrame,
    StartedGestureBotActionFrame,
    StartedMotionEffectCameraActionFrame,
    StartedPositionBotActionFrame,
    StartedPostureBotActionFrame,
    StartedShotCameraActionFrame,
    StartFacialGestureBotActionFrame,
    StartGestureBotActionFrame,
    StartMotionEffectCameraActionFrame,
    StartPositionBotActionFrame,
    StartPostureBotActionFrame,
    StartShotCameraActionFrame,
    StopFacialGestureBotActionFrame,
    StopGestureBotActionFrame,
    StopMotionEffectCameraActionFrame,
    StopPositionBotActionFrame,
    StopPostureBotActionFrame,
    StopShotCameraActionFrame,
)
from nvidia_pipecat.frames.animation import (
    AnimationDataStreamRawFrame,
    AnimationDataStreamStartedFrame,
    AnimationDataStreamStoppedFrame,
)
from nvidia_pipecat.services.action_handler import (
    ActionHandler,
    InternalStateMachineAbortFrame,
    InternalStateMachineTriggerFrame,
)
from nvidia_pipecat.services.base_action_service import (
    BaseActionService,
    ModalityManager,
    OverrideModalityManager,
    ReplaceModalityManager,
)
from nvidia_pipecat.utils.http_client import CallMethod, HttpClient
from nvidia_pipecat.utils.message_broker import MessageBrokerConfig, message_broker_factory

# Setting the number of threads is required due to an issue when running torch models in a multiprocessing context
# Since this module is imported before any multiprocess has started this ensures that
# the sentence transformer models work
# More information here: https://github.com/pytorch/pytorch/issues/36191
torch.set_num_threads(1)

# Sentence Transformers
cache_path = Path(os.getenv("ANIMATION_GRAPH_SERVICE_CACHE", "./models/"))
model_name = "all-MiniLM-L6-v2"

# We need to use the full local path here, otherwise SentenceTransformers will still query upstream models
model_cache_path = cache_path.resolve() / model_name
model_loaded_from_cache = False
if model_cache_path.is_dir():
    try:
        model = SentenceTransformer(str(model_cache_path), device="cpu")
        model_loaded_from_cache = True
    except Exception:
        # If loading from cache fails for any reason we try to redownload the model
        pass

if not model_loaded_from_cache:
    model = SentenceTransformer(model_name, device="cpu")
    try:
        model.save(str(model_cache_path))
    except Exception:
        logger.warning("Could not cache sentence transformer model. This can impact startup times.")


def _compute_embedding(text: str) -> Any:
    return model.encode(text, convert_to_tensor=True, show_progress_bar=False)


def _similarity(doc1: Any, doc2: Any) -> Any:
    return util.cos_sim(doc1, doc2)


async def _delay(coroutine, seconds) -> None:
    await asyncio.sleep(seconds)
    await coroutine


class ClipParameters(BaseModel):
    """Parameters for a clip."""

    clip_id: str
    description: str
    meaning: str
    duration: float


class AnimationConfiguration(BaseModel):
    """Configuration for an animation."""

    default_clip_id: str
    clips: list[ClipParameters]


class AnimationType(BaseModel):
    """Type of animation."""

    duration_relevant_animation_name: str
    animations: dict[str, AnimationConfiguration]


class AnimationGraphConfiguration(BaseModel):
    """Configuration for the animation graph service."""

    animation_types: dict[str, AnimationType]


@dataclass
class Animation:
    """Represents a single animation clip with metadata.

    Attributes:
        id (str): Unique identifier for the animation.
        description (str): Natural language description of the animation.
        meaning (str): Semantic meaning or purpose of the animation.
        duration (float): Length of the animation in seconds.
        description_embedding (Any): Computed embedding of the description.
        meaning_embedding (Any): Computed embedding of the meaning.
    """

    id: str
    description: str
    meaning: str
    duration: float
    description_embedding: Any
    meaning_embedding: Any

    def __str__(self) -> str:
        """Returns string representation of the animation.

        Returns:
            str: Animation ID in string format.
        """
        return f"Animation('{self.id}')"


@dataclass
class AnimationMatch:
    """Represents a match between a query and an animation.

    Attributes:
        animation (Animation): The matched animation.
        description_score (float): Similarity score for description match.
        meaning_score (float): Similarity score for meaning match.
    """

    animation: Animation
    description_score: float
    meaning_score: float


class AnimationDatabase:
    """Database for managing and querying animation clips.

    This class provides functionality to store, search, and retrieve animation clips
    based on natural language descriptions (NLD) or animation IDs. It uses semantic
    similarity to match queries against animation descriptions and meanings.

    Attributes:
        animations: List of all available animations.
        id_to_animation: Dictionary mapping animation IDs to Animation objects.
    """

    def __init__(self, available_animations: list[ClipParameters]) -> None:
        """Initializes the animation database.

        Args:
            available_animations: List of animation clip parameters to load into the database.
        """
        self.id_to_animation: dict[str, Animation] = {}
        self.animations = self._load_animations(available_animations)

    def _compute_similarities(self, nld: str) -> list[AnimationMatch]:
        query_doc = _compute_embedding(nld)
        result = []

        for anim in self.animations:
            scores = []
            for doc in [anim.description_embedding, anim.meaning_embedding]:
                scores.append(_similarity(query_doc, doc))

            result.append(AnimationMatch(anim, scores[0], scores[1]))

        return result

    def _load_animations(self, available_animations: list[ClipParameters]) -> list[Animation]:
        # Pre-compute all embeddings in batch for better performance
        descriptions = [anim.description for anim in available_animations]
        meanings = [anim.meaning for anim in available_animations]

        # Compute embeddings in batch
        description_embeddings = model.encode(descriptions, convert_to_tensor=True, show_progress_bar=False)
        meaning_embeddings = model.encode(meanings, convert_to_tensor=True, show_progress_bar=False)

        result = []
        for i, anim in enumerate(available_animations):
            a = Animation(
                anim.clip_id,
                anim.description,
                anim.meaning,
                anim.duration,
                description_embeddings[i],
                meaning_embeddings[i],
            )
            result.append(a)
            self.id_to_animation[a.id] = a
        return result

    def query(self, nld: str, n: int = 3) -> list[AnimationMatch]:
        """Query the database for animations matching a natural language description.

        Args:
            nld: Natural language description to match against.
            n: Number of top matches to return. Defaults to 3.

        Returns:
            list[AnimationMatch]: Top n animation matches by similarity score.
        """
        matches = self._compute_similarities(nld)
        sorted_matches = sorted(matches, key=lambda m: max(m.description_score, m.meaning_score))
        return sorted_matches[-n:]

    def query_one(self, nld: str) -> AnimationMatch:
        """Query the database for the best matching animation.

        Args:
            nld: Natural language description to match against.

        Returns:
            AnimationMatch: Best matching animation by similarity score.
        """
        return self.query(nld, n=1)[0]

    def query_id(self, id: str) -> Animation | None:
        """Query the database for an animation by its ID.

        Args:
            id: The ID of the animation to find.

        Returns:
            Animation | None: The matching animation if found, None otherwise.
            Case-insensitive matching is performed.
        """
        animation = self.id_to_animation.get(id, None)
        if not animation:
            for anim in self.animations:
                if anim.id.casefold().strip() == id.casefold().strip():
                    return anim
        return animation


class AnimationGraphClient(HttpClient):
    """Client for interacting with the Animation Graph service.

    This class provides methods to control various aspects of avatar animation through
    HTTP requests to the Animation Graph service. It handles state variables for
    postures, gestures, facial expressions, camera shots, and other animation controls.

    Attributes:
        url: Base URL of the Animation Graph service.
        stream_uid: Unique identifier for the animation stream.
    """

    def __init__(self, url: str, stream_uid: str) -> None:
        """Initialize the Animation Graph client.

        Args:
            url: Base URL of the Animation Graph service.
            stream_uid: Unique identifier for the animation stream.
        """
        super().__init__()
        self.url = url
        self.stream_uid = stream_uid

    async def register_stream(self) -> bool:
        """Register a new animation stream with the service.

        Returns:
            bool: True if registration was successful.
        """
        return True

    async def stop_request_playback(self, request_id: str) -> bool:
        """Stop playback of a specific animation request.

        Args:
            request_id: ID of the animation request to stop.

        Returns:
            bool: True if the playback was successfully stopped.
        """
        return await self.delete(
            url=f"{self.url}/streams/{self.stream_uid}/requests/{request_id}", headers={"x-stream-id": self.stream_uid}
        )

    async def set_state_variable(
        self, variable_name: str, variable_value: str, graph: str = "avatar", **kwargs
    ) -> bool:
        """Set a state variable in the animation graph.

        Args:
            variable_name: Name of the state variable to set.
            variable_value: Value to set for the state variable.
            graph: Name of the animation graph. Defaults to "avatar".
            **kwargs: Additional arguments to pass to the request.

        Returns:
            bool: True if the state variable was successfully set.
        """
        return await self.send_request(
            url=f"{self.url}/streams/{self.stream_uid}/animation_graphs/{graph}/variables/{variable_name}/{variable_value}",
            params={},
            payload={},
            headers={"Content-Type": "application/json", "x-stream-id": self.stream_uid},
            call_method=CallMethod.PUT,
        )

    async def set_posture_state(self, posture: str) -> bool:
        """Set the avatar's posture state.

        Args:
            posture: Name of the posture to set.

        Returns:
            bool: True if the posture state was successfully set.
        """
        return await self.set_state_variable("posture_state", posture)

    async def set_gesture_state(self, gesture: str) -> bool:
        """Set the avatar's gesture state.

        Args:
            gesture: Name of the gesture to set.

        Returns:
            bool: True if the gesture state was successfully set.
        """
        return await self.set_state_variable("gesture_state", gesture)

    async def set_facial_gesture_state(self, facial_gesture: str) -> bool:
        """Set the avatar's facial gesture state.

        Args:
            facial_gesture: Name of the facial gesture to set.

        Returns:
            bool: True if the facial gesture state was successfully set.
        """
        return await self.set_state_variable("facial_gesture_state", facial_gesture)

    async def set_position_state(self, position: str) -> bool:
        """Set the avatar's position state.

        Args:
            position: Name of the position to set.

        Returns:
            bool: True if the position state was successfully set.
        """
        return await self.set_state_variable("position_state", position)

    async def set_shot_state(self, shot: str, **kwargs) -> bool:
        """Set the camera shot state.

        Args:
            shot: Name of the camera shot to set.
            **kwargs: Additional arguments to pass to the request.

        Returns:
            bool: True if the shot state was successfully set.
        """
        return await self.set_state_variable("shot_state", shot)

    async def set_shot_transition_speed(
        self,
        start_transition: str | None = None,
        stop_transition: str | None = None,
        **kwargs,
    ) -> bool:
        """Set the speed of shot transitions.

        Args:
            start_transition: Name of the start transition effect.
            stop_transition: Name of the stop transition effect.
            **kwargs: Additional arguments to pass to the request.
        """
        if not start_transition and not stop_transition:
            return False

        value = start_transition or stop_transition or ""
        return await self.set_state_variable("shot_transition_speed", value)

    async def set_camera_motion_effect_state(self, effect: str, **kwargs) -> bool:
        """Set the camera motion effect state.

        Args:
            effect: Name of the camera motion effect to set.
            **kwargs: Additional arguments to pass to the request.

        Returns:
            bool: True if the camera motion effect state was successfully set.
        """
        return await self.set_state_variable("camera_motion_effect_state", effect)


class EmbodiedBotActionHandler(ActionHandler):
    """Base class for all action handlers bot animations.

    This includes actions like, gestures, postures, facial expressions.
    """

    _frame_type_lookup: dict[str, type[ActionFrame]] = {
        "start": ActionFrame,
        "started": ActionFrame,
        "finished": ActionFrame,
        "stop": ActionFrame,
    }

    action_name = "UnknownEmbodiedAction"
    nld_parameter_names: set[str] = set()

    def __init__(
        self,
        parent_processor: FrameProcessor,
        get_client: Callable[[], AnimationGraphClient],
        animation_databases: dict[str, AnimationDatabase],
        animation_type_config: AnimationType,
    ) -> None:
        """Initialize the embodied bot action handler.

        Args:
            parent_processor: The parent frame processor.
            get_client: Function to get the animation graph client.
            animation_databases: Dictionary of animation databases.
            animation_type_config: Configuration for this animation type.
        """
        super().__init__(parent_processor)
        self.get_client = get_client
        self.animation_databases = animation_databases
        self.machine.add_transition("timeout", source="running", dest="stopping", before=["animation_done"])  # type: ignore
        self.task: asyncio.Task | None = None
        self.resolved_parameters: dict[str, str] = {}
        self.selected_animations: dict[str, Animation] = {}
        self.previous_action_state: dict[str, Any] | None = None
        self.animation_type_config = animation_type_config
        self.nld_parameter_names = {self.animation_type_config.duration_relevant_animation_name}

    def _select_animation(self, data_base: str, animation_nld: str) -> Animation:
        """If the NLD equals an animation ID return that animation.

        Return a similarity search based match otherwise.
        """
        assert data_base in self.animation_databases

        animation = self.animation_databases[data_base].query_id(animation_nld)
        if animation:
            return animation
        else:
            return self.animation_databases[data_base].query_one(animation_nld).animation

    def _resolve_nld_parameters(self) -> None:
        self.resolved_parameters = {}
        for parameter_name, parameter_nld in self.nld_parameters.items():
            self.selected_animations[parameter_name] = self._select_animation(parameter_name, parameter_nld)
            parameter_resolved = self.selected_animations[parameter_name].id
            self.resolved_parameters[parameter_name] = parameter_resolved

        logger.info(f"{self.action_name} NLD parameter resolution: {self.resolved_parameters}")

    async def start_animation(self) -> bool:
        """Start the animation.

        Returns:
            bool: True if the animation was successfully started.
        """
        raise NotImplementedError()

    async def resume_animation(self) -> bool:
        """Resume a previously paused animation.

        Returns:
            bool: True if the animation was successfully resumed.
        """
        return await self.start_animation()

    @property
    def nld_parameters(self) -> dict[str, str]:
        """Get the natural language description parameters.

        Returns:
            dict[str, str]: Dictionary of parameter names and values.
        """
        return {name: self.action_state.get(name, "") for name in self.nld_parameter_names}

    async def on_enter_starting(self, frame: ActionFrame | InternalStateMachineTriggerFrame) -> None:
        """Handle entering the starting state.

        Args:
            frame: The frame that triggered the state transition.
        """
        self._resolve_nld_parameters()
        if not await self.start_animation():
            logger.warning("Request to Animation Graph Endpoint failed")
            await self.parent_processor.queue_for_internal_processing(
                InternalStateMachineAbortFrame(
                    action_name=self.action_name,
                    action_id=self.action_id,
                    reason="Request to Animation Graph Endpoint failed",
                )
            )
        else:
            await self.parent_processor.queue_for_internal_processing(
                self._frame_type_lookup["started"](action_id=self.action_id)
            )

    async def on_enter_stopping(self, frame: ActionFrame | InternalStateMachineTriggerFrame) -> None:
        """Handle entering the stopping state.

        Args:
            frame: The frame that triggered the state transition.
        """
        if self.task:
            await self.parent_processor.cancel_task(self.task)
            self.task = None

        await self.parent_processor.queue_for_internal_processing(
            self._frame_type_lookup["finished"](
                action_id=self.action_id,
                is_success=self.action_is_success,
                was_stopped=self.was_stopped,
                failure_reason=self.action_failure_reason,
            )
        )

    async def on_enter_paused(self, frame: ActionFrame | InternalStateMachineTriggerFrame) -> None:
        """Handle entering the paused state.

        Args:
            frame: The frame that triggered the state transition.
        """
        if self.task:
            await self.parent_processor.cancel_task(self.task)
            self.task = None

    async def on_enter_resuming(self, frame: ActionFrame | InternalStateMachineTriggerFrame) -> None:
        """Handle entering the resuming state.

        Args:
            frame: The frame that triggered the state transition.
        """
        if isinstance(frame, InternalStateMachineTriggerFrame) and "previous_action_state" in frame.data:
            self.previous_action_state = frame.data["previous_action_state"]
        if not await self.resume_animation():
            await self.parent_processor.queue_for_internal_processing(
                InternalStateMachineAbortFrame(
                    action_name=self.action_name,
                    action_id=self.action_id,
                    reason="Request to Animation Graph endpoint failed",
                )
            )
        else:
            logger.info(f"Action {self.action_name}({self.action_id}) resumed.")
            await self.parent_processor.queue_for_internal_processing(
                InternalStateMachineTriggerFrame("resumed", action_name=self.action_name, action_id=self.action_id)
            )

    async def animation_done(self, frame: ActionFrame | InternalStateMachineTriggerFrame) -> None:
        """Handle animation completion.

        Args:
            frame: The frame that triggered the completion.
        """
        self.action_is_success = True

    async def clear_modality(self) -> None:
        """Clear the current modality state."""
        await self.get_client().set_gesture_state("none")


class FiniteAnimationActionHandler(EmbodiedBotActionHandler):
    """Base class for all action handlers dealing with fixed duration animation clips."""

    _frame_type_lookup: dict[str, type[ActionFrame]] = {
        "start": ActionFrame,
        "started": ActionFrame,
        "finished": ActionFrame,
        "stop": ActionFrame,
    }
    action_name = "UnknownFiniteAnimationAction"

    # # The Animation Graph MS currently does not support sending out the end-of-clip events.
    # # therefore we take the animation duration of the following animation
    # duration_relevant_animation_name = "gesture"

    # nld_parameter_names = {duration_relevant_animation_name}
    async def on_enter_running(self, frame: ActionFrame | InternalStateMachineTriggerFrame) -> None:
        """Handle entering the running state.

        Args:
            frame: The frame that triggered the state transition.
        """
        assert self.animation_type_config.duration_relevant_animation_name in self.selected_animations

        async def animation_finished() -> None:
            params = ",".join([f"{name}={value}" for name, value in self.resolved_parameters.items()])
            logger.info(f"{self.action_name}({params}) finished. ")
            await self.parent_processor.queue_for_internal_processing(
                InternalStateMachineTriggerFrame("timeout", action_name=self.action_name, action_id=self.action_id)
            )

        seconds = timedelta(
            seconds=self.selected_animations[self.animation_type_config.duration_relevant_animation_name].duration
        ).total_seconds()
        if seconds > 0:
            self.task = self.parent_processor.create_task(_delay(animation_finished(), seconds))


class GestureBotActionHandler(FiniteAnimationActionHandler):
    """Handler for avatar gesture animations.

    Manages gesture-based animations, including starting, stopping, and transitioning
    between different gesture states.
    """

    _frame_type_lookup: dict[str, type[ActionFrame]] = {
        "start": StartGestureBotActionFrame,
        "started": StartedGestureBotActionFrame,
        "finished": FinishedGestureBotActionFrame,
        "stop": StopGestureBotActionFrame,
    }
    action_name = "GestureBotAction"

    async def start_animation(self) -> bool:
        """Start a gesture animation.

        Returns:
            bool: True if the gesture animation was successfully started.
        """
        return await self.get_client().set_gesture_state(**self.resolved_parameters)

    async def clear_modality(self) -> None:
        """Clear the current gesture state."""
        await self.get_client().set_gesture_state("none")


class FacialGestureBotActionHandler(GestureBotActionHandler):
    """Handler for avatar facial gesture animations.

    Manages facial expression animations, including starting, stopping, and transitioning
    between different facial gesture states.
    """

    _frame_type_lookup: dict[str, type[ActionFrame]] = {
        "start": StartFacialGestureBotActionFrame,
        "started": StartedFacialGestureBotActionFrame,
        "finished": FinishedFacialGestureBotActionFrame,
        "stop": StopFacialGestureBotActionFrame,
    }
    action_name = "FacialGestureBotAction"

    async def start_animation(self) -> bool:
        """Start a facial gesture animation.

        Returns:
            bool: True if the facial gesture animation was successfully started.
        """
        return await self.get_client().set_facial_gesture_state(**self.resolved_parameters)

    async def clear_modality(self) -> None:
        """Clear the current facial gesture state."""
        await self.get_client().set_facial_gesture_state("none")


class MotionEffectCameraActionHandler(FiniteAnimationActionHandler):
    """Handler for camera motion effect actions.

    Manages camera motion effects, including starting, stopping, and transitioning
    between different camera motion states.
    """

    _frame_type_lookup: dict[str, type[ActionFrame]] = {
        "start": StartMotionEffectCameraActionFrame,
        "started": StartedMotionEffectCameraActionFrame,
        "finished": FinishedMotionEffectCameraActionFrame,
        "stop": StopMotionEffectCameraActionFrame,
    }
    action_name = "MotionEffectCameraAction"

    async def start_animation(self) -> bool:
        """Start a camera motion effect.

        Returns:
            bool: True if the camera motion effect was successfully started.
        """
        return await self.get_client().set_camera_motion_effect_state(**self.resolved_parameters)

    async def clear_modality(self) -> None:
        """Clear the current camera motion effect state."""
        await self.get_client().set_camera_motion_effect_state("none")


class PostureBotActionHandler(EmbodiedBotActionHandler):
    """Handler for avatar posture animations.

    Manages posture-based animations, including starting, stopping, and transitioning
    between different posture states.
    """

    _frame_type_lookup: dict[str, type[ActionFrame]] = {
        "start": StartPostureBotActionFrame,
        "started": StartedPostureBotActionFrame,
        "finished": FinishedPostureBotActionFrame,
        "stop": StopPostureBotActionFrame,
    }
    action_name = "PostureBotAction"

    async def start_animation(self) -> bool:
        """Start a posture animation.

        Returns:
            bool: True if the posture animation was successfully started.
        """
        return await self.get_client().set_posture_state(**self.resolved_parameters)

    async def clear_modality(self) -> None:
        """Clear the current posture state."""
        await self.get_client().set_posture_state(self.animation_type_config.animations["posture"].default_clip_id)


class PositionBotActionHandler(EmbodiedBotActionHandler):
    """Handler for avatar position animations.

    Manages position-based animations, including starting, stopping, and transitioning
    between different position states.
    """

    _frame_type_lookup: dict[str, type[ActionFrame]] = {
        "start": StartPositionBotActionFrame,
        "started": StartedPositionBotActionFrame,
        "finished": FinishedPositionBotActionFrame,
        "stop": StopPositionBotActionFrame,
    }
    action_name = "PositionBotAction"

    async def start_animation(self) -> bool:
        """Start a position animation.

        Returns:
            bool: True if the position animation was successfully started.
        """
        return await self.get_client().set_position_state(**self.resolved_parameters)

    async def clear_modality(self) -> None:
        """Clear the current position state."""
        await self.get_client().set_position_state(self.animation_type_config.animations["position"].default_clip_id)


class ShotCameraActionHandler(EmbodiedBotActionHandler):
    """Handler for camera shot animations.

    Manages camera shot transitions and states, including starting, stopping, and
    transitioning between different camera angles and positions.
    """

    _frame_type_lookup: dict[str, type[ActionFrame]] = {
        "start": StartShotCameraActionFrame,
        "started": StartedShotCameraActionFrame,
        "finished": FinishedShotCameraActionFrame,
        "stop": StopShotCameraActionFrame,
    }
    action_name = "ShotCameraAction"

    async def _update_state_variables(self, **kwargs) -> bool:
        if await self.get_client().set_shot_transition_speed(**kwargs):
            return await self.get_client().set_shot_state(**kwargs)
        return False

    async def start_animation(self) -> bool:
        """Start a camera shot animation.

        Returns:
            bool: True if the camera shot animation was successfully started.
        """
        return await self._update_state_variables(**self.resolved_parameters)

    async def resume_animation(self) -> bool:
        """Resume a previously paused camera shot animation.

        Returns:
            bool: True if the camera shot animation was successfully resumed.
        """
        if self.previous_action_state:
            transition = self._select_animation("stop_transition", self.previous_action_state["stop_transition"]).id
        else:
            transition = self.resolved_parameters["start_transition"]
        parameters = {
            "shot": self.resolved_parameters["shot"],
            "start_transition": transition,
        }
        return await self._update_state_variables(**parameters)

    async def clear_modality(self) -> None:
        """Clear the current camera shot state."""
        await self.get_client().set_shot_state(self.animation_type_config.animations["shots"].default_clip_id)


def get_action_handler_factory(
    st: "AnimationGraphService.ActionConfig", service: BaseActionService, name: str
) -> Callable[[Frame], ActionHandler]:
    """Create a factory function for action handlers.

    Args:
        st: Action configuration from the AnimationGraphService.
        service: The base action service instance.
        name: Name of the action type.

    Returns:
        Callable[[Frame], ActionHandler]: Factory function that creates action handlers.
    """

    def factory(frame: Frame) -> ActionHandler:
        """Create an action handler instance.

        Args:
            frame: Frame containing action parameters.

        Returns:
            ActionHandler: New action handler instance.
        """
        return st.action_handler_cls(
            service,
            service.get_client,
            service.animation_databases,
            service.config.animation_types[name],
        )

    return factory


class AnimationGraphService(BaseActionService):
    """Manage avatar animations and interactions with the animation graph service.

    This service coordinates different types of animations including gestures, postures, facial expressions,
    and camera shots and intereacts with the animation graph service. It maintains animation databases, handles action
    state transitions, and manages animation data streaming (coming from the audio2face service).

    The service supports multiple animation types through dedicated action handlers:
        - Gesture animations (GestureBotActionHandler)
        - Posture animations (PostureBotActionHandler)
        - Facial gesture animations (FacialGestureBotActionHandler)
        - Camera shot animations (ShotCameraActionHandler)

    Each animation type has its own modality manager to handle how animations interact and override each other.
    The service processes animation frames, manages animation state, and coordinates with the animation graph client
    for streaming animation data.

    Note on startup performance:
        - To speed up the startup of the pipeline, use the class method `pregenerate_animation_databases()`
            to pre-generate the animation databases before starting the pipeline.

    Service for managing animation graphs and processing animation-related frames.

    Input Frames:
        - StartPostureBotActionFrame: Initiates a looping posture animation
        - StopPostureBotActionFrame: Stops a running posture animation

        - StartGestureBotActionFrame: Initiates a finite gesture animation
        - StopGestureBotActionFrame (optional): Stops a running gesture animation

        - StartFacialGestureBotActionFrame: Initiates a facial gesture animation
        - StopFacialGestureBotActionFrame (optional): Stops a running facial gesture animation

        - StartPositionBotActionFrame: Initiates a position animation
        - StopPositionBotActionFrame: Stops a running position animation

        - StartMotionEffectCameraActionFrame: Initiates a camera motion effect animation
        - StopMotionEffectCameraActionFrame (optional): Stops a running camera motion effect animation

        - StartShotCameraActionFrame: Initiates a camera shot animation
        - StopShotCameraActionFrame (optional): Stops a running camera shot animation

        - StartInterruptionFrame (optional): Interrupts the avatar and the current lip animation.

        - AnimationDataStreamStartedFrame (consumed): Signals start of animation data stream.
            At the moment, this is only used for lip animation data coming from the audio2face service.
        - AnimationDataStreamRawFrame (consumed): Contains animation data
        - AnimationDataStreamStoppedFrame (consumed): Signals end of animation data stream

    Output Frames:
        - StartedPostureBotActionFrame (sent up & down): Confirms posture animation has started
        - FinishedPostureBotActionFrame (sent up & down): Signals completion of posture animation

        - StartedGestureBotActionFrame (sent up & down): Confirms gesture animation has started
        - FinishedGestureBotActionFrame (sent up & down): Signals completion of gesture animation

        - StartedFacialGestureBotActionFrame (sent up & down): Confirms facial gesture animation has started
        - FinishedFacialGestureBotActionFrame (sent up & down): Signals completion of facial gesture animation

        - StartedPositionBotActionFrame (sent up & down): Confirms position animation has started
        - FinishedPositionBotActionFrame (sent up & down): Signals completion of position animation

        - StartedMotionEffectCameraActionFrame (sent up & down): Confirms camera motion effect animation has started
        - FinishedMotionEffectCameraActionFrame (sent up & down): Signals completion of camera motion effect animation

        - StartedShotCameraActionFrame (sent up & down): Confirms camera shot animation has started
        - FinishedShotCameraActionFrame (sent up & down): Signals completion of camera shot animation

        - BotStartedSpeakingFrame (sent up & down): Signals that the bot has started speaking
        - BotStoppedSpeakingFrame (sent up & down): Signals that the bot has stopped speaking
    """

    @dataclass
    class ActionConfig:
        """Configuration for an action type.

        Attributes:
            modality_manager_cls: Class to use for modality management.
            action_name: Name of the action type.
            action_handler_cls: Class to use for handling actions.
            frame_types_to_process: Types of frames this action type processes.
            failure_action_factory: Function to create failure action frames.
        """

        modality_manager_cls: type[ModalityManager]
        action_name: str
        action_handler_cls: type[EmbodiedBotActionHandler]
        frame_types_to_process: tuple[type[Frame], ...]
        failure_action_factory: Callable[[str, Frame], ActionFrame]

    supported_animation_types: dict[str, ActionConfig] = {
        "gesture": ActionConfig(
            ReplaceModalityManager,
            "GestureBotAction",
            GestureBotActionHandler,
            (
                StartGestureBotActionFrame,
                StartedGestureBotActionFrame,
                StopGestureBotActionFrame,
                FinishedGestureBotActionFrame,
            ),
            lambda reason, frame: FinishedGestureBotActionFrame(
                action_id=frame.action_id, is_success=False, failure_reason=reason
            ),
        ),
        "posture": ActionConfig(
            ReplaceModalityManager,
            "PostureBotAction",
            PostureBotActionHandler,
            (
                StartPostureBotActionFrame,
                StartedPostureBotActionFrame,
                StopPostureBotActionFrame,
                FinishedPostureBotActionFrame,
            ),
            lambda reason, frame: FinishedPostureBotActionFrame(
                action_id=frame.action_id, is_success=False, failure_reason=reason
            ),
        ),
        "facial_gesture": ActionConfig(
            ReplaceModalityManager,
            "FacialGestureBotAction",
            FacialGestureBotActionHandler,
            (
                StartFacialGestureBotActionFrame,
                StartedFacialGestureBotActionFrame,
                StopFacialGestureBotActionFrame,
                FinishedFacialGestureBotActionFrame,
            ),
            lambda reason, frame: FinishedFacialGestureBotActionFrame(
                action_id=frame.action_id, is_success=False, failure_reason=reason
            ),
        ),
        "camera_motion_effect": ActionConfig(
            ReplaceModalityManager,
            "MotionEffectCameraAction",
            MotionEffectCameraActionHandler,
            (
                StartMotionEffectCameraActionFrame,
                StartedMotionEffectCameraActionFrame,
                StopMotionEffectCameraActionFrame,
                FinishedMotionEffectCameraActionFrame,
            ),
            lambda reason, frame: FinishedMotionEffectCameraActionFrame(
                action_id=frame.action_id, is_success=False, failure_reason=reason
            ),
        ),
        "position": ActionConfig(
            OverrideModalityManager,
            "PositionBotAction",
            PositionBotActionHandler,
            (
                StartPositionBotActionFrame,
                StartedPositionBotActionFrame,
                StopPositionBotActionFrame,
                FinishedPositionBotActionFrame,
            ),
            lambda reason, frame: FinishedPositionBotActionFrame(
                action_id=frame.action_id, is_success=False, failure_reason=reason
            ),
        ),
        "camera_shot": ActionConfig(
            OverrideModalityManager,
            "ShotCameraAction",
            ShotCameraActionHandler,
            (
                StartShotCameraActionFrame,
                StartedShotCameraActionFrame,
                StopShotCameraActionFrame,
                FinishedShotCameraActionFrame,
            ),
            lambda reason, frame: FinishedShotCameraActionFrame(
                action_id=frame.action_id, is_success=False, failure_reason=reason
            ),
        ),
    }

    animation_databases: dict[str, AnimationDatabase] = {}
    frame_types_to_process: tuple[type[Frame], ...] = tuple(
        chain.from_iterable(t.frame_types_to_process for t in supported_animation_types.values())
    )

    def __init__(
        self,
        *,
        animation_graph_rest_url: str,
        animation_graph_grpc_target: str,
        message_broker_config: MessageBrokerConfig,
        config: AnimationGraphConfiguration,
        check_data_starvation: bool = True,
    ):
        """Initialize the animation graph service.

        Args:
            animation_graph_rest_url: The REST URL for the animation graph service.
            animation_graph_grpc_target: The gRPC target for the animation graph service.
            message_broker_config: The message broker configuration.
            config: The animation graph configuration.
            check_data_starvation: Whether to check for data starvation. If enabled this will print a warning as well
                as close the stream to AnimGraph if the data is not received in time. This is useful to prevent leaving
                the avatar in a bad state when e.g. the audio stream stopped too early (e.g. a TTS connectivity issue).
        """
        self.animation_graph_client: AnimationGraphClient | None = None
        self.animation_graph_rest_url = animation_graph_rest_url
        self.animation_graph_grpc_target = animation_graph_grpc_target
        self.config = config
        self.message_broker_config = message_broker_config
        self.avatar_done_talking = asyncio.Event()
        self.avatar_done_talking.set()
        self.current_speaking_action_id: str | None = None
        self._bot_speaking: bool = False

        # Data stream monitoring variables
        self._check_data_starvation = check_data_starvation
        self._data_stream_in_progress = False
        self._data_stream_warning_sent = False

        self.animation_data_queue: asyncio.Queue[
            AnimationDataStreamStartedFrame | AnimationDataStreamRawFrame | AnimationDataStreamStoppedFrame
        ] = asyncio.Queue()
        self.stream_animation_data_task: asyncio.Task | None = None
        self.process_animation_events_task: asyncio.Task | None = None
        self._bot_speaking_handler_task: asyncio.Task | None = None
        self.channel = grpc.aio.insecure_channel(self.animation_graph_grpc_target)
        self.stub = AnimationDataServiceStub(self.channel)

        managers = []
        for name in self.config.animation_types:
            if name in self.supported_animation_types:
                supported_type = self.supported_animation_types[name]

                managers.append(
                    supported_type.modality_manager_cls(
                        frame_types_to_process=supported_type.frame_types_to_process,
                        action_name=supported_type.action_name,
                        service=self,
                        action_handler_factory=get_action_handler_factory(supported_type, self, name),
                        failure_action_factory=supported_type.failure_action_factory,
                    )
                )
        super().__init__(managers)

    def get_client(self) -> AnimationGraphClient:
        """Get the animation graph client.

        Returns:
            AnimationGraphClient: The current animation graph client instance.

        Raises:
            AssertionError: If the client is not initialized.
        """
        assert self.animation_graph_client
        return self.animation_graph_client

    async def start(self, frame: StartFrame) -> None:
        """Called during pipeline start."""
        await super().start(frame)

        self.stream_animation_data_task = self.create_task(self._stream_animation_data())
        self.process_animation_events_task = self.create_task(self._process_animation_events())
        self._bot_speaking_handler_task = self.create_task(self._bot_speaking_handler())

        # Load animation databases from cache or create new ones
        for type_name, _supported_type in self.supported_animation_types.items():
            if type_name in self.config.animation_types:
                for animation_name, animation_config in self.config.animation_types[type_name].animations.items():
                    if animation_name not in AnimationGraphService.animation_databases:
                        db = AnimationDatabase(animation_config.clips)
                        AnimationGraphService.animation_databases[animation_name] = db

        await self._create_animation_graph_client()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process incoming frames.

        Handles animation data stream frames and manages animation state transitions.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        if isinstance(
            frame, AnimationDataStreamStartedFrame | AnimationDataStreamRawFrame | AnimationDataStreamStoppedFrame
        ):
            # Put into animation data queue and don't push AnimationDataStream frames any further
            self.animation_data_queue.put_nowait(frame)
        else:
            await super().process_frame(frame, direction)
            if isinstance(frame, StartInterruptionFrame):
                await self._interrupt_avatar()

    async def _interrupt_avatar(self) -> None:
        logger.debug("_interrupt_avatar called")

        if self.current_speaking_action_id:
            logger.debug(f"stopping animation clip with action_id={self.current_speaking_action_id}")
            if not await self.get_client().stop_request_playback(self.current_speaking_action_id):
                # This can often happen if the playback finished before the stop request was received
                # so we don't want to it as an error, but just log it as a debug message
                info_message = f"Stopping playback for {self.current_speaking_action_id} failed (usually harmless)"
                logger.debug(info_message)
                # await self.push_frame(ErrorFrame(error_message))
        else:
            logger.debug("received StartInterruptionFrame when no speaking animation clip is playing")

        await self._bot_stopped_speaking()
        logger.debug("waiting for streaming task to finish")
        await self._stop_stream_animation_data_task()
        logger.debug("creating new data streaming task with an empty queue")
        self.animation_data_queue = asyncio.Queue()
        self.stream_animation_data_task = self.create_task(self._stream_animation_data())

    async def _process_animation_events(self):
        """Subscribe to animation graph events redis pub/sub channel to receive events about animation clip playback."""
        try:
            broker = message_broker_factory(config=self.message_broker_config, channels=[])
            await broker.wait_for_connection()
        except Exception as e:
            logger.exception(f"Could not create message broker {e}")
            return

        while True:
            try:
                message = await broker.pubsub_receive_message(channels=["animation_graph_events"], timeout=None)
                # Check if got a message and that it is for the current stream before loading it (which is expensive)
                if message and self.stream_id in message:
                    logger.debug(f"received event from animation graph ms: {message}")
                    event = json.loads(message)
                    if (
                        event["event_type"] == "request_playback_ended"
                        or event["event_type"] == "request_playback_interrupted"
                    ):
                        logger.debug(
                            f"Bot stopped speaking based on event received from animation graph: {event['event_type']}"
                        )
                        await self._bot_stopped_speaking()
            except Exception as e:
                logger.error(e)

    async def _bot_started_speaking(self):
        self.avatar_done_talking.clear()
        if not self._bot_speaking:
            logger.debug("Bot started speaking")
            await self.push_frame(BotStartedSpeakingFrame())
            await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            self._bot_speaking = True

    async def _bot_stopped_speaking(self):
        self.avatar_done_talking.set()
        if self._bot_speaking:
            logger.debug("Bot stopped speaking")
            await self.push_frame(BotStoppedSpeakingFrame())
            await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            self._bot_speaking = False

    async def _bot_speaking_handler(self):
        """This task sends out BotSpeakingFrames every 200ms to fulfill the protocol of the output transports.

        We have no way of knowing when exactly audio chunks are being sent by the renderer to the client
        so we simply send a BotSpeakingFrame every 200ms.
        """
        TIMEOUT = 0.2
        while not self._cancelling:
            await asyncio.sleep(TIMEOUT)
            if self._bot_speaking:
                await self.push_frame(BotSpeakingFrame())
                await self.push_frame(BotSpeakingFrame(), FrameDirection.UPSTREAM)

    async def _stream_animation_data(self) -> None:
        stream: StreamUnaryCall[AnimationDataStream, Status] | None = None
        DATA_STREAM_TIMEOUT = 0.3
        while not self._cancelling:
            try:
                if stream:
                    frame = await asyncio.wait_for(self.animation_data_queue.get(), timeout=DATA_STREAM_TIMEOUT)
                else:
                    frame = await self.animation_data_queue.get()

                new_message = None

                if isinstance(frame, AnimationDataStreamStartedFrame):
                    await self.avatar_done_talking.wait()
                    stream = self.stub.PushAnimationDataStream(metadata=(("x-stream-id", self.stream_id),))
                    audio_header: AudioHeader = frame.audio_header
                    self.current_speaking_action_id = frame.action_id
                    logger.debug(f"Sending AnimationData header with request_id={self.current_speaking_action_id}")
                    new_message = AnimationDataStream(
                        animation_data_stream_header=AnimationDataStreamHeader(
                            animation_ids=AnimationIds(
                                stream_id=self.stream_id,
                                request_id=self.current_speaking_action_id,
                                target_object_id="toto",
                            ),
                            source_service_id=frame.animation_source_id,
                            audio_header=audio_header,
                            skel_animation_header=frame.animation_header,
                            start_time_code_since_epoch=time.time(),
                        )
                    )
                    await self._bot_started_speaking()

                elif isinstance(frame, AnimationDataStreamRawFrame):
                    new_message = AnimationDataStream(
                        animation_data=frame.animation_data,
                    )
                elif isinstance(frame, AnimationDataStreamStoppedFrame):
                    logger.debug(
                        f"AnimationData stream stopped for request_id={self.current_speaking_action_id}. "
                        "Does not affect playback."
                    )
                    if stream and not stream.done():
                        await stream.done_writing()
                        stream = None

                # Send new message only if we have an active stream
                if new_message and stream and not stream.done():
                    await stream.write(new_message)

                if await self._check_datastream_starvation(frame):
                    await self._close_stream(stream)
                    stream = None

            except TimeoutError:
                if await self._check_datastream_starvation(None):
                    await self._close_stream(stream)
                    stream = None
            except Exception as e:
                logger.error(f"Exception: {e}")

    async def _create_animation_graph_client(self) -> None:
        if self.stream_id:
            await self._close_animation_graph_client()
            self.animation_graph_client = AnimationGraphClient(
                self.animation_graph_rest_url,
                self.stream_id,
            )
            await self.animation_graph_client.register_stream()

    async def _close_animation_graph_client(self) -> None:
        if self.animation_graph_client:
            await self.animation_graph_client.close()

    async def _stop_stream_animation_data_task(self) -> None:
        if self.stream_animation_data_task:
            await self.cancel_task(self.stream_animation_data_task, timeout=0.3)
            self.stream_animation_data_task = None
            self._data_stream_in_progress = False

    async def _stop_running_tasks(self) -> None:
        await self._stop_stream_animation_data_task()
        if self.process_animation_events_task:
            await self.cancel_task(self.process_animation_events_task, timeout=0.3)
            self.process_animation_events_task = None
        if self._bot_speaking_handler_task:
            await self.cancel_task(self._bot_speaking_handler_task, timeout=0.3)
            self._bot_speaking_handler_task = None

    async def cleanup(self) -> None:
        """Clean up the service."""
        await super().cleanup()
        await self._close_animation_graph_client()
        await self._stop_running_tasks()

    async def stop(self, frame: EndFrame) -> None:
        """Called during pipeline end."""
        await self._close_animation_graph_client()
        await self._stop_running_tasks()

    @classmethod
    def pregenerate_animation_databases(cls, config: AnimationGraphConfiguration) -> None:
        """Pre-generate animation databases from configuration and cache them.

        You can do this before you start the pipeline you don't have to wait
        for the databases to be created during startup.

        Args:
            config: Animation graph configuration containing animation types and clips

        """
        # Generate databases for each animation type
        for type_name in config.animation_types:
            for animation_name, animation_config in config.animation_types[type_name].animations.items():
                db = AnimationDatabase(animation_config.clips)
                cls.animation_databases[animation_name] = db

    async def _close_stream(self, stream: StreamUnaryCall[AnimationDataStream, Status]) -> None:
        logger.debug(
            f"Closing stream for request_id={self.current_speaking_action_id}"
            " before StoppedFrame was received due to stream starvation."
        )
        self._data_stream_in_progress = False
        if stream and not stream.done():
            await stream.done_writing()
            try:
                await self.get_client().stop_request_playback(self.current_speaking_action_id)
            except Exception as e:
                logger.info(f"Could not stop request playback on timeout: {e}")

    async def _check_datastream_starvation(
        self,
        frame: AnimationDataStreamStartedFrame | AnimationDataStreamRawFrame | AnimationDataStreamStoppedFrame | None,
    ) -> bool:
        if not self._check_data_starvation:
            return False

        WARNING_TIMEOUT_S = 0.1  # Print warning if data is delayed by more than this
        STARVATION_CLOSE_TIMEOUT_S = 1.0  # Indicate starvation if data is delayed by more than this

        if isinstance(frame, AnimationDataStreamStartedFrame):
            self._data_until_playback_starves = time.monotonic()
            self._audio_samples_per_second = frame.audio_header.samples_per_second
            self._audio_bits_per_sample = frame.audio_header.bits_per_sample
            self._audio_channel_count = frame.audio_header.channel_count
            self._data_stream_in_progress = True
            self._data_stream_warning_sent = False

        elif isinstance(frame, AnimationDataStreamStoppedFrame):
            self._data_stream_in_progress = False

        elif self._data_stream_in_progress and isinstance(frame, AnimationDataStreamRawFrame):
            audio_buffer_size = len(frame.animation_data.audio.audio_buffer)

            audio_buffer_size_s = audio_buffer_size / (
                self._audio_samples_per_second * self._audio_bits_per_sample / 8 * self._audio_channel_count
            )

            self._data_until_playback_starves += audio_buffer_size_s

        if self._data_stream_in_progress:
            now = time.monotonic()
            if now > self._data_until_playback_starves + STARVATION_CLOSE_TIMEOUT_S:
                logger.warning(
                    f"Data stream starvation detected: data behind by {now - self._data_until_playback_starves}s"
                )
                await self.push_frame(
                    ErrorFrame("AnimGraph: Data stream starvation detected. AnimGraph connection will be reset.")
                )
                return True

            elif now > self._data_until_playback_starves + WARNING_TIMEOUT_S:
                logger.info(f"Data stream data behind by {now - self._data_until_playback_starves}s")
                if not self._data_stream_warning_sent:
                    await self.push_frame(
                        ErrorFrame(f"AnimGraph: Received data stream is behind by more than {WARNING_TIMEOUT_S}s")
                    )
                    self._data_stream_warning_sent = True
                return False
        else:
            return False
