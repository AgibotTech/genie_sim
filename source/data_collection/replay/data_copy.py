"""
After recording, copy source data into each replay output directory.
Exclude captured media and recording_info.json to preserve the new versions.
"""
import os
import shutil

from common.base_utils.logger import logger


def _default_ignore(_src: str, names: list[str]) -> list[str]:
    """Exclude captured media and metadata to preserve the new recording."""
    exclude_names = {"observation", "observations", "camera", "recording_info.json"}
    return [n for n in names if n in exclude_names]


def copy_original_to_recording(
    src_dir: str,
    dst_dir: str,
) -> None:
    """
    Copy source data from the state.json directory into the output recording.
    Exclude observation, observations, camera, and recording_info.json.
    """
    src_dir = os.path.normpath(src_dir.rstrip("/"))
    dst_dir = os.path.normpath(dst_dir.rstrip("/"))
    if not os.path.isdir(src_dir):
        logger.warning("copy_original_to_recording: src not a dir %s", src_dir)
        return
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, name)
        dst_path = os.path.join(dst_dir, name)
        if name in _default_ignore(src_dir, [name]):
            continue
        try:
            if os.path.isdir(src_path):
                if os.path.exists(dst_path):
                    shutil.copytree(
                        src_path,
                        dst_path,
                        ignore=_default_ignore,
                        dirs_exist_ok=True,
                    )
                else:
                    shutil.copytree(src_path, dst_path, ignore=_default_ignore)
            else:
                shutil.copy2(src_path, dst_path)
        except Exception as e:
            logger.warning("copy_original_to_recording: failed %s -> %s: %s", src_path, dst_path, e)
    logger.info("copy_original_to_recording: done %s -> %s", src_dir, dst_dir)
