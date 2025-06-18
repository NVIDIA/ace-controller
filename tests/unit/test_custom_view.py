# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the CustomView related frames."""

import json

from nvidia_pipecat.frames.custom_view import (
    Button,
    ButtonListBlock,
    ButtonVariant,
    HeaderBlock,
    Hint,
    HintCarouselBlock,
    Image,
    ImageBlock,
    ImagePosition,
    ImageWithTextBlock,
    SelectableOption,
    SelectableOptionsGridBlock,
    StartCustomViewFrame,
    TableBlock,
    TextBlock,
    TextInputBlock,
)


def test_to_json_empty():
    """Tests empty custom view JSON serialization.

    Tests:
        - Empty frame conversion
        - Minimal configuration
        - Default values

    Raises:
        AssertionError: If JSON output doesn't match expected format.
    """
    frame = StartCustomViewFrame(action_id="test-action-id")
    result = frame.to_json()
    expected_result = {}
    assert result == json.dumps(expected_result)


def test_to_json_simple():
    """Tests simple custom view JSON serialization.

    Tests:
        - Basic block configuration
        - Header block formatting
        - Single block conversion

    Raises:
        AssertionError: If JSON output doesn't match expected format.
    """
    frame = StartCustomViewFrame(
        action_id="test-action-id",
        blocks=[
            HeaderBlock(id="test-header", header="Test Header", level=1),
        ],
    )
    result = frame.to_json()
    expected_result = {
        "blocks": [{"id": "test-header", "type": "header", "data": {"header": "Test Header", "level": 1}}],
    }
    assert result == json.dumps(expected_result)


def test_to_json_complex():
    """Tests complex custom view JSON serialization.

    Tests:
        - Multiple block types
        - Image handling
        - Nested data structures

    Raises:
        AssertionError: If JSON output doesn't match expected format.
    """
    # Sample base64 string for image data
    sample_base64 = "iVBORw0KGgoAAAANSUhEUgAAA5UAAAC5CAIAAAA3TIxUAADd3"

    frame = StartCustomViewFrame(
        action_id="test-action-id",
        blocks=[
            HeaderBlock(id="test-header", header="Test Header", level=1),
            TextBlock(id="test-text", text="Test Text"),
            ImageBlock(id="test-image", image=Image(url="https://example.com/image.jpg")),
            ImageWithTextBlock(
                id="test-image-with-text",
                image=Image(data=sample_base64),
                text="Test Text",
                image_position=ImagePosition.LEFT,
            ),
            TableBlock(id="test-table", headers=["Header 1", "Header 2"], rows=[["Row 1", "Row 2"]]),
            HintCarouselBlock(
                id="test-hint-carousel", hints=[Hint(id="test-hint", name="Test Hint", text="Test Text")]
            ),
            ButtonListBlock(
                id="test-button-list",
                buttons=[
                    Button(
                        id="test-button",
                        active=True,
                        toggled=False,
                        variant=ButtonVariant.CONTAINED,
                        text="Test Button",
                    )
                ],
            ),
            SelectableOptionsGridBlock(
                id="test-selectable-options-grid",
                buttons=[
                    SelectableOption(
                        id="test-option",
                        image=Image(url="https://example.com/image.jpg"),
                        text="Test Option",
                        active=True,
                        toggled=False,
                    )
                ],
            ),
            TextInputBlock(
                id="test-input",
                default_value="Test Default Value",
                value="Test Value",
                label="Test Label",
                input_type="text",
            ),
        ],
    )
    result = frame.to_json()
    expected_result = {
        "blocks": [
            {"id": "test-header", "type": "header", "data": {"header": "Test Header", "level": 1}},
            {"id": "test-text", "type": "paragraph", "data": {"text": "Test Text"}},
            {
                "id": "test-image",
                "type": "image",
                "data": {"image": {"url": "https://example.com/image.jpg"}},
            },
            {
                "id": "test-image-with-text",
                "type": "paragraph_with_image",
                "data": {
                    "image": {"data": sample_base64},
                    "text": "Test Text",
                    "image_position": "left",
                },
            },
            {
                "id": "test-table",
                "type": "table",
                "data": {"headers": ["Header 1", "Header 2"], "rows": [["Row 1", "Row 2"]]},
            },
            {
                "id": "test-hint-carousel",
                "type": "hint_carousel",
                "data": {"hints": [{"name": "Test Hint", "text": "Test Text"}]},
            },
            {
                "id": "test-button-list",
                "type": "button_list",
                "data": {
                    "buttons": [
                        {
                            "id": "test-button",
                            "active": True,
                            "toggled": False,
                            "variant": "contained",
                            "text": "Test Button",
                        }
                    ]
                },
            },
            {
                "id": "test-selectable-options-grid",
                "type": "selectable_options_grid",
                "data": {
                    "buttons": [
                        {
                            "id": "test-option",
                            "image": {"url": "https://example.com/image.jpg"},
                            "text": "Test Option",
                            "active": True,
                            "toggled": False,
                        }
                    ]
                },
            },
            {
                "id": "test-input",
                "type": "text_input",
                "data": {
                    "default_value": "Test Default Value",
                    "value": "Test Value",
                    "label": "Test Label",
                    "input_type": "text",
                },
            },
        ],
    }
    assert result == json.dumps(expected_result)
