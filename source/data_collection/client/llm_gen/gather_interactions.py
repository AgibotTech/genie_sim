import os
import sys
import importlib

sim_assets = os.environ.get("SIM_ASSETS")
if not sim_assets:
    raise RuntimeError("SIM_ASSETS environment variable is not set")

# SIM_ASSETS is a package directory with __init__.py; its name may differ from assets.
# Add its parent to sys.path and import the actual directory name dynamically.
sim_assets = os.path.abspath(sim_assets)
sim_assets_parent = os.path.dirname(sim_assets)
sim_assets_pkg = os.path.basename(sim_assets)
sys.path.insert(0, sim_assets_parent)

assets_mod = importlib.import_module(sim_assets_pkg)
try:
    ASSETS_INDEX = assets_mod.ASSETS_INDEX
except AttributeError as e:
    raise AttributeError(f"ASSETS_INDEX not found in {sim_assets}; ensure __init__.py exports it") from e

def gather_interactions(assets_ids: list[str], just_structure: bool = False) -> dict:
    def _keep_only_labels(section: dict) -> dict:
        """Recursively preserve interaction labels and structure without parameter values.
        For example, {"place": {"side": [...], "upright": [...]}} becomes
        {"place": {"side": {}, "upright": {}}}."""
        if not isinstance(section, dict):
            return {}
        result = {}
        for k, v in section.items():
            if isinstance(v, dict):
                # Recurse into nested dictionaries.
                result[k] = _keep_only_labels(v)
            else:
                # Replace concrete values such as lists and strings with empty dictionaries.
                result[k] = ["..."]
        return result

    interactions = {}
    for asset_id in assets_ids:
        asset_info = ASSETS_INDEX.get(asset_id, {})
        if not asset_info:
            print(f"Asset {asset_id} not found in ASSETS_INDEX")
            continue
        asset_interaction = asset_info.get("interaction", {})
        if not asset_interaction:
            print(f"Interaction for asset {asset_id} not found")
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

if __name__ == "__main__":
    demo_assets_ids = [
        "benchmark_storage_box_006",
        "benchmark_storage_box_007",
        "benchmark_storage_box_008",
        "benchmark_storage_box_009",
        "benchmark_storage_box_010",
        "benchmark_apple_000",
        "benchmark_orange_001",
        "benchmark_peach_000",
        "benchmark_lemon_027",
    ]
    demo_interactions = gather_interactions(demo_assets_ids, just_structure=True)
    print(demo_interactions)