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
embedding_mode = os.getenv("EMBEDDING_MODE", "vl")
use_reranker = os.getenv("USE_RERANKER", "True") == "True"
permanent_model_in_gpu = os.getenv("PERMANENT_MODEL_IN_GPU", "False") == "True"
if embedding_mode == "vl":
    from geniesim.generator.server.assets_searcher.assets_searcher_vl import AssetVectorDBVL
else:
    from geniesim.generator.server.assets_searcher.assets_searcher import AssetVectorDB
from fastmcp import FastMCP
from typing import List, Dict
from pydantic import BaseModel, Field
from typing import Annotated

# Configure logging to use stderr to avoid interfering with JSON-RPC on stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Assets Agent Server")


if embedding_mode == "vl":
    config = {
        "model": os.getenv("VL_EMBEDDING_MODEL", "Qwen3-VL-Embedding-2B"),
        "use_reranker": use_reranker,
        "permanent_model_in_gpu": permanent_model_in_gpu,
    }
    db = AssetVectorDBVL(config=config)
else:
    config = json.load(open(f"{CURRENT_DIRECTORY}/mcp_text_embedding/text_embedding_config.json"))
    db = AssetVectorDB(config=config)


@mcp.tool(name="search_assets")
def search_assets(
    keyword: Annotated[
        str,
        Field(
            ...,
            description="Keyword used to search for assets, the more detailed the keyword, the more accurate the search results",
        ),
    ],
    topk: Annotated[int, Field(10, description="Number of assets to return, default is 10")],
    exclude_regex: Annotated[
        str,
        Field(
            None,
            description="Regular expression pattern to match asset_id. Assets with matching asset_id will be excluded from search results. Use '|' for OR logic. Examples: 'omni6D', 'omni6D.*', 'omni6D.*|stf.*' (exclude assets starting with 'omni6D' or 'stf'). Default is None (no exclusion).",
            examples=["omni6D", "omni6D.*", "omni6D.*|sft.*"],
        ),
    ],
    include_regex: Annotated[
        str,
        Field(
            None,
            description="Regular expression pattern to match asset_id. Assets with matching asset_id will be included in search results. Use '|' for OR logic. Examples: 'omni6D', 'omni6D.*', 'omni6D.*|stf.*' (include assets starting with 'omni6D' or 'stf'). Default is None (no inclusion).",
            examples=["omni6D", "omni6D.*", "omni6D.*|sft.*"],
        ),
    ],
    scene_description: Annotated[
        str,
        Field(
            "",
            description="The summary of the scene description to search for assets, it will help the search agent to search for more relevant assets, default is empty string",
            examples=["There are some bottles of sweet soda on the table"],
        ),
    ],
) -> List[Dict]:
    """
    Search for assets by keyword

    Args:

        keyword: Keyword used to search for assets.
        topk: Number of assets to return, default is 10.
        exclude_regex: Regular expression pattern to match asset_id. Assets with matching asset_id will be excluded from search results. Use '|' for OR logic. Examples: 'omni6D', 'omni6D.*', 'omni6D.*|stf.*' (exclude assets starting with 'omni6D' or 'stf'). Default is None (no exclusion).
        include_regex: Regular expression pattern to match asset_id. Assets with matching asset_id will be included in search results. Use '|' for OR logic. Examples: 'omni6D', 'omni6D.*', 'omni6D.*|stf.*' (include assets starting with 'omni6D' or 'stf'). Default is None (no inclusion).
        scene_description: The summary of the scene description to search for assets, it will help the search agent to search for more relevant assets, default is empty string

    Returns:

        List containing asset information, each asset includes asset_id, semantic_name full_description and other fields.
    """
    try:
        logger.info(f"Searching assets: {keyword}, topk: {topk}")
        results = db.search(keyword, topk, exclude_regex, include_regex, scene_description)
        logger.info(f"Search results: {len(results)} assets found")
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        error_msg = f"Error searching assets: {str(e)}"
        raise RuntimeError(error_msg) from e


@mcp.tool(name="search_asset_by_asset_id")
def search_asset_by_asset_id(asset_id: Annotated[str, Field(..., description="Asset ID to search for")]) -> List[Dict]:
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
