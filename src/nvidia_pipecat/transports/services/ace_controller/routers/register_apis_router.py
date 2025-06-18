# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Add and Remove Pipeline HTTP APIs for ACE Controller Server for Supporting RTSP SDR calls.

Need to be registered in the fastapi app and need ACEPipelineRunner to be initialized.
"""

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field

from nvidia_pipecat.pipeline.ace_pipeline_runner import ACEPipelineRunner

router = APIRouter()


class StreamEvent(BaseModel):
    """Schema for event for stream registration."""

    camera_url: str = Field("", description="RTSP URL of the stream")
    camera_id: str = Field(..., description="Unique identifier for the stream")


class StreamRequest(BaseModel):
    """Schema for request for stream registration."""

    event: StreamEvent


@router.post("/stream/add")
async def add_stream(request: StreamRequest):
    """Register a new pipeline / stream ID.

    Args:
        request: StreamRequest object containing stream registration details.
            Schema:
            {
                "event": {
                    "camera_url": str,  # RTSP URL of the stream
                    "camera_id": str    # Unique identifier for the stream
                }
            }

    Returns:
        dict: A dictionary with a message indicating the successful addition of the stream ID.
    """
    rtsp_url = request.event.camera_url
    stream_id = request.event.camera_id
    logger.info(f"Received request to add stream ID {stream_id} with RTSP URL {rtsp_url}")
    await ACEPipelineRunner.get_instance().add_pipeline(stream_id, rtsp_url)
    logger.info(f"Successfully added stream ID {stream_id}")
    return {"message": f"Stream ID {stream_id} added"}


@router.post("/stream/remove")
async def remove_stream(request: StreamRequest):
    """Remove a pipeline / stream ID.

    Args:
        request: StreamRequest object containing stream removal details.
            Schema:
            {
                "event": {
                    "camera_id": str    # ID of the stream to remove
                }
            }

    Returns:
        dict: A dictionary with a message indicating the successful removal of the stream ID.
    """
    stream_id = request.event.camera_id
    logger.info(f"Received request to remove stream ID {stream_id}")
    await ACEPipelineRunner.get_instance().remove_pipeline(stream_id)
    logger.info(f"Successfully removed stream ID {stream_id}")
    return {"message": f"Stream ID {stream_id} removed"}
