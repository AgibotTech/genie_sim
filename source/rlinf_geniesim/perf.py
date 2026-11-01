# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""CPU timing traces enabled by GENIESIM_PERF_LOG=1; rows stream to disk."""

import csv
import os
from pathlib import Path
import tempfile
import time


class PerfLogger:
    def __init__(self):
        self._fp = None
        self.count = 0
        self._writer = None
        self.path = None
        # One file per handler, so reloads in the same process do not truncate.
        if os.environ.get("GENIESIM_PERF_LOG", "0") == "1":
            self._fp = tempfile.NamedTemporaryFile(
                mode="w",
                prefix=f"rlinf_perf_{os.getpid()}_",
                suffix=".csv",
                dir=tempfile.gettempdir(),
                delete=False,
                buffering=1,
            )
            self.path = Path(self._fp.name)

    def log(self, **row):
        if self._fp is None:
            return
        self.count += 1
        row = {"record": self.count, "monotonic_s": time.monotonic(), **row}
        if self._writer is None:
            self._writer = csv.DictWriter(self._fp, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None
