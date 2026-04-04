import os
import sys
import logging

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)
sys.path.append(ROOT_DIRECTORY)

from fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import Annotated
from geniesim.assets import ASSETS_INDEX

# Configure logging to use stderr to avoid interfering with JSON-RPC on stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Assets Info Agent Server")


def gather_interactions(assets_ids: list[str], just_structure: bool = True) -> dict:
    def _keep_only_labels(section: dict) -> dict:
        """
        Recursively keep only interaction structure / labels, discarding concrete parameter values.
        Example:
        {"place": {"side": [...], "upright": [...]}}
        -> {"place": {"side": {}, "upright": {}}}
        """
        if not isinstance(section, dict):
            return {}
        result = {}
        for k, v in section.items():
            if isinstance(v, dict):
                # Values that are dicts: recurse
                result[k] = _keep_only_labels(v)
            else:
                # Lists, strings, etc.: replace with placeholder
                result[k] = ["..."]
        return result

    interactions = {}
    for asset_id in assets_ids:
        asset_info = ASSETS_INDEX.get(asset_id, {})
        if not asset_info:
            logger.info(f"Asset {asset_id} not found in ASSETS_INDEX")
            continue
        asset_interaction = asset_info.get("interaction", {})
        if not asset_interaction:
            logger.info(f"Interaction for asset {asset_id} not found")
            continue
        if just_structure:
            active = asset_interaction.get("active", {})
            passive = asset_interaction.get("passive", {})
            interactions[asset_id] = {
                "active": _keep_only_labels(active),
                "passive": _keep_only_labels(passive),
            }
        else:
            interactions[asset_id] = asset_interaction
    return interactions


@mcp.tool(name="get_interactions")
def get_interactions(
    assets_ids: Annotated[list[str], Field(..., description="List of asset IDs to get interactions for")],
    just_structure: Annotated[
        bool,
        Field(
            True,
            description="Whether to return only the structure of the interactions without any real values, default is True",
        ),
    ],
) -> dict:
    """
    Get interactions for a list of asset IDs
    Args:
        assets_ids: List of asset IDs to get interactions for
        just_structure: Whether to return only the structure of the interactions without any real values, default is True
    Returns:
        Dictionary containing the interactions for the given asset IDs
    """
    try:
        logger.info(f"Getting interactions for assets: {assets_ids}, just_structure: {just_structure}")
        interactions = gather_interactions(assets_ids, just_structure)
        return interactions
    except Exception as e:
        logger.error(f"Error getting interactions: {e}")
        error_msg = f"Error getting interactions: {str(e)}"
        return {"error": error_msg}


if __name__ == "__main__":
    mcp.run(transport="stdio")
