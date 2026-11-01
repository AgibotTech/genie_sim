from __future__ import annotations

import os
import sys
from pathlib import Path


def prepare_cosmos_environment(cosmos_repo: Path):
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "UV_OFFLINE": "1"})
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if os.environ.get("_G01_NVIDIA_LIBS_READY"):
        return
    library_dirs = []
    for site in Path(sys.prefix).glob("lib/python*/site-packages"):
        library_dirs.extend(
            path for path in (site / "nvidia").glob("**/lib") if any(path.glob("lib*.so*"))
        )
    if not library_dirs:
        raise RuntimeError("Use the Cosmos Framework venv with NVIDIA runtime libraries installed")
    os.environ["LD_LIBRARY_PATH"] = ":".join(map(str, library_dirs)) + (
        ":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""
    )
    os.environ.update(
        {
            "_G01_NVIDIA_LIBS_READY": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "UV_OFFLINE": "1",
        }
    )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.execv(sys.executable, [sys.executable, "-m", "geniesim_dream", *sys.argv[1:]])
