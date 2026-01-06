# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import sys
import logging

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)
sys.path.append(ROOT_DIRECTORY)

from fastmcp import FastMCP

# Configure logging to use stderr to avoid interfering with JSON-RPC on stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("File Agent Server")


@mcp.tool()
def save_file(file_path: str, content: str) -> bool:
    """
    Save JSON file by path

    Args:
        file_path: File path
        content: Content to write, in JSON format

    Returns:
        Returns whether save was successful: Bool
    """
    file_path_dir = os.path.dirname(file_path)
    if not os.path.exists(file_path_dir):
        os.makedirs(file_path_dir)
    try:
        logger.info(f"Saving file: {file_path}")
        with open(file_path, "w") as f:
            f.write(content)
            return True
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return False


if __name__ == "__main__":
    mcp.run(transport="stdio")
