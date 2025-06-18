# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""WebSocket endpoints for ACE Controller."""

from fastapi import APIRouter, WebSocket
from loguru import logger

from nvidia_pipecat.pipeline.ace_pipeline_runner import ACEPipelineRunner

router = APIRouter()

"""
WebSocket endpoints for ACE Controller Server

Need to be registered in the fastapi app and need ACEPipelineRunner to be initialized.
"""


@router.websocket("/ws/{stream_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    stream_id: str,
):
    """Accept the WebSocket connection and update the pipeline manager.

    Args:
        websocket (WebSocket): The WebSocket connection.
        stream_id (str): The ID of the stream.
    """
    # Accept the WebSocket connection.
    logger.info(f"Accepting WebSocket connection for stream ID {stream_id}")
    await websocket.accept()
    try:
        # Update the pipeline with the websocket connection.
        await ACEPipelineRunner.get_instance().connect_websocket(stream_id, websocket)
    except ValueError as e:
        logger.error(f"Error updating pipeline: {str(e)}")
        await websocket.close(code=1000, reason=str(e))
    except Exception as e:
        logger.error(f"Error updating pipeline: {e}")
        await websocket.close(code=1000, reason="Internal Server Error")
