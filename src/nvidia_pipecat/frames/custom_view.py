# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Custom view frames."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from pipecat.frames.frames import DataFrame
from pydantic import BaseModel, Field

from nvidia_pipecat.frames.action import (
    StartActionFrame,
    StopActionFrame,
    UserActionFrame,
)


# Block types
class Block(BaseModel, ABC):
    """A block is a modular component that can be rendered in the UI.

    Multiple blocks can be rendered at a time.

    Args:
        id (str): A unique id for the block.
    """

    id: str

    @abstractmethod
    def get_type(self) -> str:
        """Get the type of the block.

        Type indicates what data is included in the block and is used to determine how to render the block in the UI.
        """


class HeaderBlock(Block):
    """A header block displays a header or subheader in the UI.

    Args:
        header (str): The text to display in the header.
        level (int): The level of the header, corresponding to HTML element tags for header (h1, h2, etc.).
            Must be between 1 and 6.
    """

    header: str
    level: int = Field(ge=1, le=6)

    def get_type(self) -> str:
        """Get the type of the block."""
        return "header"


class TextBlock(Block):
    """A text block displays text as a paragraph in the UI.

    Args:
        text (str): The text to display in the paragraph.
    """

    text: str

    def get_type(self) -> str:
        """Get the type of the block."""
        return "paragraph"


class Image(BaseModel):
    """Specifies the data needed to render an image. file and data fields are mutually exclusive.

    Args:
        url (str | None): The URL of the image.
        data (str | None): The image base64 data.
    """

    url: str | None = None
    data: str | None = None


class ImageBlock(Block):
    """An image block displays an image in the UI.

    Args:
        image (Image): The image to display.
        caption (str | None): The caption to display in place of the image if the image cannot be rendered.
    """

    image: Image
    caption: str | None = None

    def get_type(self) -> str:
        """Get the type of the block."""
        return "image"


class ImagePosition(str, Enum):
    """Enum for image position."""

    LEFT = "left"
    RIGHT = "right"


class ImageWithTextBlock(Block):
    """An image with text block displays an image and text in the UI.

    Args:
        image (Image): The image to display.
        text (str): The text to display next to the image.
        image_position (ImagePosition): The position of the image relative to the text.
    """

    image: Image
    text: str
    image_position: ImagePosition

    def get_type(self) -> str:
        """Get the type of the block."""
        return "paragraph_with_image"


class TableBlock(Block):
    """A table block displays a table in the UI.

    Args:
        headers (list[str]): The headers of the table.
        rows (list[list[str]]): The rows of the table.
    """

    headers: list[str]
    rows: list[list[str]]

    def get_type(self) -> str:
        """Get the type of the block."""
        return "table"


class Hint(BaseModel):
    """A hint is an example of how the user can interact with the system.

    Args:
        name (str): The name of the hint. This is used to identify the hint in the UI.
        text (str): The text of the hint. The text is what is displayed to the user.
    """

    name: str
    text: str


class HintCarouselBlock(Block):
    """A hint carousel block.

    The block rotates through a list of hints, displaying
    them one at a time to the user.

    Args:
        hints (list[Hint]): A list of hints to display.
    """

    hints: list[Hint]

    def get_type(self) -> str:
        """Get the type of the block."""
        return "hint_carousel"


class ButtonVariant(str, Enum):
    """Enum for button variant styles."""

    OUTLINED = "outlined"
    CONTAINED = "contained"
    TEXT = "text"


class Button(BaseModel):
    """Data format with information about a button.

    Args:
        id (str): The id of the button.
        active (bool): Whether the button is active. If false, the button will be grayed out and not clickable.
        toggled (bool): Whether the button is toggled. If true, the button will be toggled on.
        variant (ButtonVariant): The variant of the button.
        text (str): The text of the button.
    """

    id: str
    active: bool
    toggled: bool
    variant: ButtonVariant
    text: str


class ButtonListBlock(Block):
    """A button list block displays a list of buttons.

    Args:
        buttons (list[Button]): A list of buttons to display.
    """

    buttons: list[Button]

    def get_type(self) -> str:
        """Get the type of the block."""
        return "button_list"


class SelectableOption(BaseModel):
    """Data format for a selectable option. Can include an image and/or text.

    Args:
        id (str): The id of the selectable option.
        image (Image | None): The image to display within the option.
        text (str): The text to display within the option.
        active (bool): Whether the option is active. If false, the option will be grayed out and not clickable.
        toggled (bool): Whether the option is toggled. If true, the option will be toggled on.
    """

    id: str
    image: Image | None = None
    text: str
    active: bool
    toggled: bool


class SelectableOptionsGridBlock(Block):
    """A selectable options grid block.

    This block displays a grid of interactive options to the user.
    A user may select and deselect any number of options as they choose.

    Args:
        options (list[SelectableOption]): A list of selectable options to display.
    """

    buttons: list[SelectableOption]

    def get_type(self) -> str:
        """Get the type of the block."""
        return "selectable_options_grid"


class TextInputBlock(Block):
    """A text input block displays a text input to the user.

    Args:
        id (str): The id of the text input.
        default_value (str): The default value of the text input.
        value (str): The value of the text input.
        label (str): The label of the text input.
        input_type (str): The type of the text input.
    """

    id: str
    default_value: str
    value: str
    label: str
    input_type: str

    def get_type(self) -> str:
        """Get the type of the block."""
        return "text_input"


@dataclass
class StartCustomViewFrame(StartActionFrame):
    """Sends data to the UI in the form of modular components to be rendered.

    Args:
        blocks (list[Block] | None): A list of modular components to display in the UI.
        style (Style | None): The style of the UI.
    """

    blocks: list[Block] | None = None

    def to_json(self) -> str:
        """Convert the frame to a JSON string.

        This is intended to be used for serializing the frame for transmission over a WebSocket connection to the UI.
        """
        the_json = {}
        if self.blocks:
            blocks_list_json = []
            for block in self.blocks:
                block_json = {
                    "id": block.id,
                    "type": block.get_type(),
                    "data": block.model_dump(exclude={"id"}, exclude_none=True),
                }
                blocks_list_json.append(block_json)
            the_json["blocks"] = blocks_list_json
        return json.dumps(the_json)


@dataclass
class StopCustomViewFrame(StopActionFrame):
    """A frame that stops the custom view from being rendered in the UI."""


@dataclass
class UIInterimTextInputFrame(UserActionFrame, DataFrame):
    """Interim text input frame from the UI.

    Receives data from the UI in the form of text inputted into a textbox
    by the user prior to submission.

    Args:
        component_id (str): The id of the text input which is currently being typed in.
        interim_input (str): The interim input of the text input.
    """

    component_id: str
    interim_input: str


@dataclass
class UITextInputFrame(UserActionFrame, DataFrame):
    """A frame that receives data from the UI in the form of a text inputted into a textbox.

    This is triggered when the user submits their input.

    Args:
        enter_pressed (bool): True if the enter key was pressed to submit the text input,
            false if the user navigated away from the text input.
        component_id (str): The id of the text input which was submitted.
        input (str): The input of the text input.
    """

    enter_pressed: bool
    component_id: str
    input: str


@dataclass
class UIButtonPressFrame(UserActionFrame, DataFrame):
    """A frame that receives data from the UI in the form of a button press.

    Args:
        component_id (str): The id of the button which was pressed.
    """

    component_id: str


@dataclass
class UISelectableOptionPressFrame(UserActionFrame, DataFrame):
    """A frame that receives data from the UI in the form of a selectable option press.

    Args:
        component_id (str): The id of the selectable option which was pressed.
        toggled (bool): Whether the option is toggled. If true, the option will be toggled on.
    """

    component_id: str
    toggled: bool
