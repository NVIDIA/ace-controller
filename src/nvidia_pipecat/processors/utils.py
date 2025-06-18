"""Utility processors for pipecat."""

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class FrameBlockingProcessor(FrameProcessor):
    """A frame processor that blocks forwarding frames after a specified frame number of a specific type.

    This processor counts frames of a specific type and stops forwarding them after reaching a specified threshold.
    It can be used to limit the number of specific frame types processed in a pipeline.

    Args:
        block_after_frame (int): The frame number after which to block forwarding frames.
        frame_type (Type[Frame]): The type of frame to count and block.
        reset_frame_type (Optional[Type[Frame]]): If provided, frames of this type will reset the counter.
        **kwargs: Additional arguments passed to parent FrameProcessor.
    """

    def __init__(
        self, block_after_frame: int, frame_type: type[Frame], reset_frame_type: type[Frame] | None = None, **kwargs
    ):
        """Initialize the frame blocking processor.

        Args:
            block_after_frame: The frame number after which to block forwarding frames.
            frame_type: The type of frame to count and block.
            reset_frame_type: If provided, frames of this type will reset the counter.
            **kwargs: Additional arguments passed to parent FrameProcessor.
        """
        super().__init__(**kwargs)
        self.block_after_frame = block_after_frame
        self.frame_type = frame_type
        self.reset_frame_type = reset_frame_type
        self.frame_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process a frame.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        # Check for reset frame type first
        if self.reset_frame_type and isinstance(frame, self.reset_frame_type):
            logger.debug(f"Resetting counter on {self.reset_frame_type.__name__} frame")
            self.frame_count = 0
            await self.push_frame(frame, direction)
        elif isinstance(frame, self.frame_type):
            self.frame_count += 1
            if self.frame_count <= self.block_after_frame:
                await self.push_frame(frame, direction)
            else:
                logger.debug(
                    f"Blocking {self.frame_type.__name__} frame {self.frame_count}"
                    f" (threshold: {self.block_after_frame})"
                )
        else:
            # Forward non-matching frame types without counting
            await self.push_frame(frame, direction)
