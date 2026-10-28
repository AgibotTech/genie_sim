# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""CPU frame timing; streaming output, one file per stage generation."""
import csv
import os
import tempfile


class RenderPerfLogger:
    def __init__(self):
        self._fp = None
        self._writer = None
        if os.environ.get("GENIESIM_RENDER_PERF_LOG", "1") != "0":
            self._fp = tempfile.NamedTemporaryFile(
                mode="w", prefix=f"rlinf_render_perf_{os.getpid()}_",
                suffix=".csv", dir=tempfile.gettempdir(),
                buffering=1, delete=False)
            print(f"[render-perf] CPU timings: {self._fp.name}", flush=True)

    def log(self, **row):
        if self._fp is None:
            return
        if self._writer is None:
            self._writer = csv.DictWriter(self._fp, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None
