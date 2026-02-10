# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
Token counting utilities for OpenAI API requests with text and images.
"""

import math
from typing import Dict, List, Optional
import numpy as np


def calculate_text_tokens(text: str) -> int:
    """
    Calculate approximate token count for text.
    Uses a simple approximation: ~4 characters per token for English text.

    Args:
        text: Input text string

    Returns:
        Estimated token count
    """
    # Simple approximation: ~4 characters per token
    # This is a rough estimate, for exact count would need tiktoken
    return math.ceil(len(text) / 4)


def calculate_input_tokens(
    content: List[dict], image_history: Optional[List[Dict[str, np.ndarray]]] = None
) -> Dict[str, int]:
    """
    Calculate text token count for the content list.

    Args:
        content: List of content items (text and image_url)
        image_history: Deprecated, kept for backward compatibility (ignored)

    Returns:
        Dictionary with token breakdown: {
            'text_tokens': int,
            'image_tokens': int (always 0),
            'total_tokens': int (same as text_tokens),
            'image_count': int
        }
    """
    text_tokens = 0
    image_count = 0

    # Calculate text tokens from content
    for item in content:
        if item.get("type") == "text":
            text = item.get("text", "")
            text_tokens += calculate_text_tokens(text)
        elif item.get("type") == "image_url":
            image_count += 1

    return {
        "text_tokens": text_tokens,
        "total_tokens": text_tokens,
        "image_count": image_count,
    }
