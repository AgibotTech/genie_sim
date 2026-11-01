import os
from pickle import TRUE
import sys
import random
from pathlib import Path

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
from common.base_utils.logger import logger
import argparse

from scripts.resolve_llm_task_root import resolve_llm_task_root
from scripts.llm_task_policy import validate_run_times

DEFAULT_TASK_ROOT_DIR = str(resolve_llm_task_root(Path(root_path)) or Path(root_path) / "llm_tasks_data_collection")

def change_payload(scene_usda_path: str) -> None:
    """
    Replace the asset root in scene.usda payload paths with SIM_ASSETS.
    
    Find the assets directory preceding Aligned.usda and replace that prefix with SIM_ASSETS.
    
    Use a file lock to serialize updates across processes.
    
    Args:
        scene_usda_path: Path to scene.usda.
    """
    import re
    import fcntl
    
    # Read the asset root from SIM_ASSETS.
    sim_assets = os.environ.get("SIM_ASSETS")
    if not sim_assets:
        raise ValueError("Environment variable SIM_ASSETS is not set")
    sim_assets = sim_assets.rstrip("/")
    
    # Protect file operations with a lock.
    lock_file_path = scene_usda_path + ".lock"
    
    try:
        # Create and acquire the lock.
        with open(lock_file_path, 'w', encoding='utf-8') as lock_file:
            # Try to acquire an exclusive lock without blocking.
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except IOError:
                # If another process holds the lock, wait until it is released.
                logger.info(f"File {scene_usda_path} is locked, waiting...")
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)  # Wait for the lock.
            
            # Read the file contents.
            with open(scene_usda_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            def _replace_asset_path(match):
                path = match.group(1)
                normalized_path = path.replace('\\', '/')
                normalized_sim_assets = sim_assets.replace('\\', '/')
                if normalized_path.startswith(normalized_sim_assets + '/') or normalized_path == normalized_sim_assets:
                    return match.group(0)
                assets_pattern = r'assets[/\\]+'
                assets_match = re.search(assets_pattern, path, re.IGNORECASE)
                if assets_match:
                    assets_part = path[assets_match.end():].replace('\\', '/').lstrip('/')
                    new_path = f"{sim_assets}/{assets_part}"
                    logger.info(f"Replaced path: {path} -> {new_path}")
                    return f"@{new_path}@"
                return match.group(0)

            new_content = re.sub(r'@([^@]+)@', _replace_asset_path, content)
            
            if new_content != content:
                temp_file_path = scene_usda_path + ".tmp"
                try:
                    with open(temp_file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    os.replace(temp_file_path, scene_usda_path)
                    logger.info(f"Updated paths in {scene_usda_path}")
                except Exception:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    raise
            else:
                logger.info(f"No paths to update in {scene_usda_path}")
    finally:
        # Remove the lock file.
        if os.path.exists(lock_file_path):
            try:
                os.remove(lock_file_path)
            except OSError:
                pass


def get_scene_dir(task_name: str, scene_dir: str = "", root_dir: str = DEFAULT_TASK_ROOT_DIR):
    target_dir = os.path.join(root_dir, task_name)
    candidates = []
    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)
        if os.path.isdir(item_path):
            # item_path must contain scene_info.json and scene.usda.
            scene_info_json = os.path.join(item_path, "scene_info.json")
            scene_usda = os.path.join(item_path, "scene.usda")
            if os.path.isfile(scene_info_json) and os.path.isfile(scene_usda):
                candidates.append(os.path.abspath(item_path))
    if not candidates:
        raise RuntimeError(f"No numeric scene directories found in {target_dir}")
    
    if scene_dir == "random":
        return random.choice(candidates)
    
    # Try to match scene_dir.
    if scene_dir:
        matching_dir = os.path.join(target_dir, scene_dir)
        if os.path.isdir(matching_dir):
            return matching_dir
    
    return candidates[0]


def run_llm_task(
    task_root_dir: str = DEFAULT_TASK_ROOT_DIR,
    task_name: str = "table_with_cubes_by_size",
    scene_dir: str = "random",
    run_times: int = 1,
    use_recording: bool = False,
    client_host: str = "localhost:50051",
):
    validate_run_times(Path(task_root_dir) / task_name, scene_dir, run_times)
    from client.llm_gen.robot_api import RobotAPIConfig, RobotAPI
    path_success_dict = []
    robot = None
    for i in range(run_times):
        chosen_scene_dir = get_scene_dir(task_name, scene_dir, task_root_dir)
        logger.info(f"Loading task {chosen_scene_dir}")
        change_payload(os.path.join(chosen_scene_dir, "scene.usda"))
        task_info_json = os.path.join(task_root_dir, task_name, "task_info.json")
        config = RobotAPIConfig.from_json(task_info_json)
        config.auto_record = use_recording
        robot = RobotAPI(
            config=config,
            scene_usd=os.path.join(chosen_scene_dir, "scene.usda"),
            scene_info=os.path.join(chosen_scene_dir, "scene_info.json"),
            client_host=client_host,
            from_robot=robot,
        )
        sys.path.append(os.path.join(task_root_dir, task_name))
        from data_gen import execute_task
        try:
            success = execute_task(robot)
        except Exception as e:
            logger.error(f"Error executing task: {e}", exc_info=True)
            success = False
        robot.stop_recording()
        robot.client.send_task_status(success, [robot.stage_id, -1])
        path_success_dict.append({"path": chosen_scene_dir, "success": success})
    robot.exit()
    logger.info(path_success_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--task_name", type=str, default="desk_with_trash_and_stationery")
    parser.add_argument("--task_root_dir", type=str, default=DEFAULT_TASK_ROOT_DIR)
    parser.add_argument("--task_name", type=str, default="pack_supermarket_arx")
    parser.add_argument("--scene_dir", type=str, default="random", help="scene directory name or 'random'")
    parser.add_argument("--run_times", type=int, default=1, help="run times")
    parser.add_argument("--use_recording", default=False, action="store_true", help="use recording")
    parser.add_argument("--client_host", type=str, default="localhost:50051", help="client host")
    args = parser.parse_args()
    task_name = args.task_name
    scene_dir = args.scene_dir
    run_times = args.run_times
    run_llm_task(task_root_dir=args.task_root_dir, task_name=args.task_name, scene_dir=args.scene_dir, run_times=args.run_times, use_recording=args.use_recording, client_host=args.client_host)