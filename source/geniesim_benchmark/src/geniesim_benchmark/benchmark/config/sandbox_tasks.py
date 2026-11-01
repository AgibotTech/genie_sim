# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""The 25 open-world tasks migrated from the pinned demo release."""

import json
from pathlib import Path

SANDBOX_TASKS = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))["tasks"]
