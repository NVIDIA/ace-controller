# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Pipeline runner for ACE."""

import asyncio
import gc
from dataclasses import dataclass

from fastapi import WebSocket
from loguru import logger
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from starlette.websockets import WebSocketState


@dataclass
class PipelineMetadata:
    """Metadata for managing pipeline state and connections.

    This class holds the necessary information to track and manage a pipeline instance,
    including its stream ID, websocket connection, RTSP URL, and associated tasks.

    Attributes:
        stream_id: Unique identifier for the pipeline stream
        websocket: Optional WebSocket connection for the pipeline
        rtsp_url: RTSP URL for video/audio streaming
        pipeline_task: Task object representing the pipeline process
        runner_task: Asyncio task managing the pipeline execution
    """

    stream_id: str
    websocket: WebSocket | None = None
    rtsp_url: str = ""
    pipeline_task: PipelineTask | None = None
    runner_task: asyncio.Task | None = None


class ACEPipelineRunner:
    """Singleton class for managing ACE pipelines.

    This class provides a singleton interface for managing multiple ACE pipelines,
    including the addition of new pipelines, connection of websockets, and removal of pipelines.

    Attributes:
        _pipelines: Dictionary storing pipeline metadata for each stream ID
        _enable_rtsp: Boolean flag indicating if RTSP is enabled
        _pipelines_callback: Callback function for pipeline creation
        __instance: Singleton instance of the class
    """

    __instance = None

    def __init__(self, pipeline_callback: callable, enable_rtsp: bool = False):
        """Initialize the ACEPipelineRunner singleton.

        Args:
            pipeline_callback: Callback function for pipeline creation
            enable_rtsp: Boolean flag indicating if RTSP is enabled
        """
        if ACEPipelineRunner.__instance is not None:
            raise Exception("This class is a singleton!")
        self._pipelines = {}
        self._enable_rtsp = enable_rtsp
        self._pipelines_callback = pipeline_callback
        ACEPipelineRunner.__instance = self

    @staticmethod
    def create_instance(pipeline_callback: callable, enable_rtsp: bool = False):
        """Create an instance of the ACEPipelineRunner.

        Args:
            pipeline_callback: Callback function for pipeline creation
            enable_rtsp: Boolean flag indicating if RTSP is enabled
        """
        if ACEPipelineRunner.__instance is not None:
            return ACEPipelineRunner.__instance
        else:
            ACEPipelineRunner.__instance = ACEPipelineRunner(pipeline_callback, enable_rtsp)
            return ACEPipelineRunner.__instance

    @staticmethod
    def get_instance():
        """Get the singleton instance of the ACEPipelineRunner.

        Returns:
            ACEPipelineRunner: The singleton instance of the class

        Raises:
            Exception: If the class is not initialized
        """
        if ACEPipelineRunner.__instance is None:
            raise Exception("This class is a singleton!, Please create an instance first.")
        return ACEPipelineRunner.__instance

    async def add_pipeline(self, stream_id: str, rtsp_url: str):
        """Add a new pipeline to the runner.

        Args:
            stream_id: Unique identifier for the pipeline stream
            rtsp_url: RTSP URL string for video/audio streaming
        """
        logger.debug(f"Found {len(self._pipelines)} active pipelines with stream IDs: {self._pipelines.keys()}")
        with logger.contextualize(stream_id=stream_id):
            if stream_id in self._pipelines:
                raise ValueError(f"Pipeline for Stream ID {stream_id} already exists")

            self._pipelines[stream_id] = PipelineMetadata(stream_id, rtsp_url=rtsp_url)
            logger.info(f"Received add pipeline request, Running pipeline for stream {stream_id}")
            try:
                await self._run_pipeline(stream_id)
            except Exception as e:
                logger.error(f"Error while creating pipeline: {e}")
                await self.remove_pipeline(stream_id)
                raise ValueError(f"Error while creating pipeline: {e}") from e

    async def connect_websocket(self, stream_id: str, websocket: WebSocket):
        """Connect a websocket.

        Connects a websocket to the running pipeline or creates a new pipeline if
        it is not running for the given stream id. The method will wait until the websocket
        connection is closed by the client and only then return.

        Args:
            stream_id: Unique identifier for the pipeline stream
            websocket: WebSocket connection for the pipeline
        """
        with logger.contextualize(stream_id=stream_id):
            # First check if pipeline exists
            if self._enable_rtsp and stream_id not in self._pipelines:
                raise ValueError(f"Pipeline for Stream ID {stream_id} does not exist")
            elif not self._enable_rtsp and stream_id not in self._pipelines:
                self._pipelines[stream_id] = PipelineMetadata(stream_id, websocket=websocket)
                await self._run_pipeline(stream_id)
                await self._wait_for_websocket_close(stream_id)
                await self.remove_pipeline(stream_id)
            elif self._pipelines[stream_id].runner_task and self._pipelines[stream_id].runner_task.done():
                raise ValueError(f"Pipeline for Stream ID {stream_id} is already terminated")
            else:
                await self._update_websocket(stream_id, websocket)
                await self._wait_for_websocket_close(stream_id)

    async def remove_pipeline(self, stream_id: str):
        """Remove a pipeline from the runner.

        Args:
            stream_id: Unique identifier for the pipeline stream
        """
        with logger.contextualize(stream_id=stream_id):
            if stream_id in self._pipelines:
                if self._pipelines[stream_id].pipeline_task is not None:
                    try:
                        # Signal shutdown
                        await self._pipelines[stream_id].pipeline_task.stop_when_done()
                    except Exception as e:
                        logger.error(f"Error while removing Pipeline: {e}")

                    if self._pipelines[stream_id].runner_task and not self._pipelines[stream_id].runner_task.done():
                        logger.info("Waiting for pipeline runner task to finish ...")
                        await self._pipelines[stream_id].runner_task
                logger.info(f"Pipeline for Stream ID {stream_id} removed")

    async def _cleanup_pipeline(self, stream_id: str):
        """Cleanup a pipeline.

        Args:
            stream_id: Unique identifier for the pipeline stream
        """
        try:
            if (
                stream_id in self._pipelines
                and self._pipelines[stream_id].websocket
                and self._pipelines[stream_id].websocket.client_state == WebSocketState.CONNECTED
            ):
                await self._pipelines[stream_id].websocket.close()
        except Exception as e:
            logger.error(f"Error while closing websocket: {e}")
        del self._pipelines[stream_id]
        gc.collect()
        logger.info(f"Pipeline for Stream ID {stream_id} deleted")

    async def _run_pipeline(self, stream_id: str):
        """Run a pipeline in background.

        Args:
            stream_id: Unique identifier for the pipeline stream
        """
        try:
            self._pipelines[stream_id].pipeline_task = await self._pipelines_callback(self._pipelines[stream_id])
            self._pipelines[stream_id].pipeline_task.set_event_loop(asyncio.get_event_loop())
            self._pipelines[stream_id].runner_task = asyncio.create_task(self._pipelines[stream_id].pipeline_task.run())
            self._pipelines[stream_id].runner_task.add_done_callback(
                lambda _: asyncio.create_task(self._cleanup_pipeline(stream_id))
            )
            logger.info(f"Pipeline started successfully for stream {stream_id}")
        except Exception as e:
            logger.error(f"Error while creating pipeline task: {e}")
            raise

    async def _update_websocket(self, stream_id: str, websocket: WebSocket):
        """Update the websocket for a pipeline.

        Args:
            stream_id: Unique identifier for the pipeline stream
            websocket: WebSocket connection for the pipeline
        """
        self._pipelines[stream_id].websocket = websocket
        pipeline = self._pipelines[stream_id].pipeline_task._pipeline
        for component in pipeline._processors:
            if isinstance(component, BaseInputTransport | BaseOutputTransport):
                if hasattr(component._transport, "update_websocket"):
                    await component._transport.update_websocket(websocket)
                    logger.info(f"Websocket for Stream ID {stream_id} updated")
                    return
                else:
                    raise ValueError(f"Component {component.__class__.__name__} doesn't support updating websocket.")

    async def _wait_for_websocket_close(self, stream_id: str):
        """Wait for the websocket to close. This is used to keep connection alive.

        Args:
            stream_id: Unique identifier for the pipeline stream
        """
        try:
            # Wait until websocket is closed
            while (
                stream_id in self._pipelines
                and self._pipelines[stream_id].websocket.client_state == WebSocketState.CONNECTED
            ):
                await asyncio.sleep(0.1)
            logger.info(f"Websocket for Stream ID {stream_id} closed")
        except Exception as e:
            raise ValueError(f"Error while waiting for websocket close: {e}") from e
