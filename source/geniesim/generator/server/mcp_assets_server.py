# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import sys
import logging
import json

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)
sys.path.append(ROOT_DIRECTORY)

from geniesim.generator.server.assets_searcher import AssetVectorDB
from fastmcp import FastMCP
from typing import List, Dict

# Configure logging to use stderr to avoid interfering with JSON-RPC on stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Assets Agent Server")

# Initialize asset database
config = json.load(open(f"{CURRENT_DIRECTORY}/text_embedding_config.json"))
db = AssetVectorDB(config=config)


@mcp.tool()
def search_assets(keyword: str, topk: int = 10) -> List[Dict]:
    """
    Search for assets by keyword

    Args:
        keyword: Keyword used to search for assets
        topk: Number of assets to return, default is 10

    Returns:
        List containing asset information, each asset includes asset_id, semantic_name,
        full_description and other fields
    """
    try:
        logger.info(f"Searching assets: {keyword}, topk: {topk}")
        results = db.search(keyword, topk)
        logger.info(f"Search results: {len(results)} assets found")
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        error_msg = f"Error searching assets: {str(e)}"
        raise RuntimeError(error_msg) from e


@mcp.tool()
def search_asset_by_asset_id(asset_id: str) -> List[Dict]:
    """
    Search for asset by asset ID (requires exact ID match), returns one asset or empty list if not found
    Args:
        asset_id: Asset ID
    Returns:
        List containing asset information, each asset includes asset_id, semantic_name,
        full_description and other fields
    """
    try:
        logger.info(f"Searching asset: {asset_id}")
        results = db.search_by_asset_id(asset_id)
        logger.info(f"Search results: {len(results)} assets found")
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        error_msg = f"Error searching assets: {str(e)}"
        raise RuntimeError(error_msg) from e


if __name__ == "__main__":
    mcp.run(transport="stdio")
