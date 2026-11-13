# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import urlencode
from urllib.request import urlopen

APP_ROOT = Path(__file__).resolve().parent / "app"
REVISION = "37ef0dbcda6be7b23a352560339589bf2b8e5488"
SHA256 = "7e748611278a354e1acdfd0b502e778f0cb5e1f6313d2f65fd29fc9f659443b8"
SIZE = 18368638
URL = "https://www.modelscope.cn/api/v1/datasets/agibot_world/GenieSimAssets/repo?" + urlencode(
    {"Source": "SDK", "Revision": REVISION, "FilePath": "runtime/teleop/app.tar.gz", "View": False}
)
REQUIRED_FILES = (
    "bin/genie_motion_control",
    "bin/start_mc.sh",
    "bin/env.sh",
    "vendors/lib/libpinocchio_casadi.so.3.7.0",
    "configuration/robot/G2/model/default.yaml",
    "share/genie_robot_description/urdf/G2/G2.urdf",
    "share/geniesim_msg/package.xml",
    "share/ros_plugin_msgs/package.xml",
)


def app_ready(root):
    return all((root / name).is_file() for name in REQUIRED_FILES) and os.access(
        root / "bin/genie_motion_control", os.X_OK
    )


def archive_valid(path):
    if not path.is_file() or path.stat().st_size != SIZE:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == SHA256


def ensure_app(app_root=APP_ROOT):
    root = Path(app_root).absolute()
    print(f"Checking local teleop app: {root}", flush=True)
    if app_ready(root):
        print("Local teleop app found; skipping download.", flush=True)
        return root
    print("Local teleop app is missing or incomplete; preparing the runtime package.", flush=True)
    cache = root.parent / ".cache/teleop-runtime"
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if app_ready(root):
            print("Teleop app is ready from another process; skipping download.", flush=True)
            return root
        archive = cache / "app.tar.gz"
        with tempfile.TemporaryDirectory(prefix="install-", dir=cache) as directory:
            temporary = Path(directory)
            if not archive_valid(archive):
                print(
                    f"Downloading only runtime/teleop/app.tar.gz ({SIZE / 1024**2:.2f} MiB) "
                    "from ModelScope agibot_world/GenieSimAssets ...",
                    flush=True,
                )
                partial = temporary / "app.tar.gz"
                with urlopen(URL, timeout=60) as response, partial.open("wb") as output:
                    size = 0
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        size += len(block)
                        if size > SIZE:
                            raise RuntimeError("Teleop runtime download exceeds the expected size")
                        output.write(block)
                print("Download complete; verifying SHA256 ...", flush=True)
                if not archive_valid(partial):
                    raise RuntimeError("Teleop runtime download failed SHA256 verification")
                partial.replace(archive)
            else:
                print("Cached app.tar.gz passed SHA256 verification; skipping download.", flush=True)
            # Only extract the immutable archive verified above.
            print("SHA256 verified; extracting the teleop app ...", flush=True)
            subprocess.run(["tar", "-xzf", str(archive), "-C", str(temporary)], check=True)
            extracted = temporary / "app"
            if not app_ready(extracted):
                raise RuntimeError("Teleop runtime archive is incomplete")
            previous = temporary / "previous-app"
            if root.exists() or root.is_symlink():
                root.rename(previous)
            try:
                extracted.rename(root)
            except OSError:
                if previous.exists() or previous.is_symlink():
                    previous.rename(root)
                raise
        print(f"Teleop runtime ready: {root}", flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description="Download the motion-control runtime when it is missing.")
    parser.add_argument("--app-dir", type=Path, default=APP_ROOT)
    args = parser.parse_args()
    ensure_app(args.app_dir)


if __name__ == "__main__":
    main()
