# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import asyncio
import json
import aiofiles
from typing import Dict, Any, Optional
from pathlib import Path
import logging


class AsyncJSONWriter:
    """
    Async JSON writer: accepts data continuously and writes to files asynchronously.
    """

    def __init__(
        self,
        max_queue_size: int = 1000,
        batch_size: int = 10,
        flush_interval: float = 1.0,
    ):
        """
        Initialize the async JSON writer.

        Args:
            max_queue_size: max queue size to avoid memory overflow
            flush_interval: flush interval (seconds) for periodic writes
        """
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # data queue: (data, file_path)
        self.data_queue = asyncio.Queue(maxsize=max_queue_size)

        # control
        self._is_running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._pending_writes: Dict[str, dict] = {}  # file_path -> data

        # stats
        self.total_written = 0
        self._logger = logging.getLogger(__name__)

    async def start(self):
        """Start the writer."""
        if self._is_running:
            return

        self._is_running = True
        self._worker_task = asyncio.create_task(self._write_worker())
        self._logger.info("AsyncJSONWriter started")

    async def stop(self):
        """Stop the writer and wait for all data to be written."""
        if not self._is_running:
            return

        self._is_running = False

        # wait for worker to finish
        if self._worker_task and not self._worker_task.done():
            await self._worker_task

        # flush all pending writes
        await self._flush_all()

        self._logger.info(f"AsyncJSONWriter stopped, total written: {self.total_written}")

    async def add_data(self, data: Dict[str, Any], file_path: str):
        """
        Add data to be written.

        Args:
            data: dict to write
            file_path: target file path
        """
        if not self._is_running:
            raise RuntimeError("Writer not started; call start() first")

        try:
            # non-blocking put; wait if queue full
            await self.data_queue.put((data, file_path))
        except asyncio.QueueFull:
            self._logger.warning("Data queue full, waiting for space...")
            await self.data_queue.put((data, file_path))

    async def _write_worker(self):
        """Worker coroutine: process write tasks continuously."""
        last_flush_time = asyncio.get_event_loop().time()
        while self._is_running or not self.data_queue.empty():
            try:
                try:
                    # wait for data or timeout (for periodic flush)
                    data, file_path = await asyncio.wait_for(self.data_queue.get(), timeout=self.flush_interval)
                    self._pending_writes[file_path] = data
                    if len(self._pending_writes) >= self.batch_size:
                        await self._flush_all()

                except asyncio.TimeoutError:
                    pass

                # periodic flush
                current_time = asyncio.get_event_loop().time()
                if current_time - last_flush_time >= self.flush_interval:
                    await self._flush_all()
                    last_flush_time = current_time

            except Exception as e:
                self._logger.error(f"Write worker error: {e}")
                await asyncio.sleep(0.1)  # avoid tight error loop

    async def _flush_file(self, file_path):
        """Flush pending data for the given file."""
        if file_path not in self._pending_writes or not self._pending_writes[file_path]:
            return
        try:
            data = self._pending_writes.pop(file_path)
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
                json_str = json.dumps(
                    data,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    check_circular=False,
                )
                await f.write(json_str)
                self.total_written += 1

        except Exception as e:
            self._logger.error(f"Failed to write file {file_path}: {e}")
            # keep data for retry

    async def _flush_all(self):
        """Flush pending data for all files."""
        for file_path in list(self._pending_writes.keys()):
            if self._pending_writes[file_path]:
                await self._flush_file(file_path)

    def get_stats(self) -> Dict[str, Any]:
        """Return stats."""
        return {
            "is_running": self._is_running,
            "total_written": self.total_written,
            "queue_size": self.data_queue.qsize(),
            "files_pending": list(self._pending_writes.keys()),
        }
