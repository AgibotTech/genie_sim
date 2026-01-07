# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles

from common.base_utils.logger import logger


class AsyncJSONWriter:
    """
    Async JSON writer that can continuously receive data and automatically write to files asynchronously
    """

    def __init__(
        self,
        max_queue_size: int = 1000,
        batch_size: int = 10,
        flush_interval: float = 1.0,
    ):
        """
        Initialize async JSON writer

        Args:
            max_queue_size: Maximum queue size to prevent memory overflow
            flush_interval: Flush interval (seconds), periodically write data in queue
        """
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # Data queue: (data, file_path)
        self.data_queue = asyncio.Queue(maxsize=max_queue_size)

        # Control variables
        self._is_running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._pending_writes: Dict[str, dict] = {}  # file_path -> [data_list]

        # Statistics
        self.total_written = 0
        # Use global logger instance

    async def start(self):
        """Start writer"""
        if self._is_running:
            return

        self._is_running = True
        self._worker_task = asyncio.create_task(self._write_worker())
        logger.info("AsyncJSONWriter started")

    async def stop(self):
        """Stop writer, wait for all data to be written"""
        if not self._is_running:
            return

        self._is_running = False

        # Wait for worker task to complete
        if self._worker_task and not self._worker_task.done():
            await self._worker_task

        # Force flush all pending data
        await self._flush_all()

        logger.info(f"AsyncJSONWriter stopped, total {self.total_written} records written")

    async def add_data(self, data: Dict[str, Any], file_path: str):
        """
        Add data to be written

        Args:
            data: Dictionary data to write
            file_path: Target file path
        """
        if not self._is_running:
            raise RuntimeError("Writer not started, please call start() method first")

        try:
            # Add data in non-blocking way, wait if queue is full
            await self.data_queue.put((data, file_path))
        except asyncio.QueueFull:
            logger.warning("Data queue is full, waiting for space...")
            await self.data_queue.put((data, file_path))

    async def _write_worker(self):
        """Worker coroutine, continuously process write tasks"""
        last_flush_time = asyncio.get_event_loop().time()
        while self._is_running or not self.data_queue.empty():
            try:
                # Wait for data or timeout
                try:
                    # Set timeout, periodically flush data
                    data, file_path = await asyncio.wait_for(
                        self.data_queue.get(), timeout=self.flush_interval
                    )
                    # Accumulate pending write data
                    self._pending_writes[file_path] = data
                    if len(self._pending_writes) >= self.batch_size:
                        # If too many files pending, prioritize flushing
                        await self._flush_all()

                except asyncio.TimeoutError:
                    # Timeout, check if flush is needed
                    pass

                # Periodically flush all files
                current_time = asyncio.get_event_loop().time()
                if current_time - last_flush_time >= self.flush_interval:
                    await self._flush_all()
                    last_flush_time = current_time

            except Exception as e:
                logger.error(f"Write worker error: {e}")
                await asyncio.sleep(0.1)  # Prevent error loop

    async def _flush_file(self, file_path):
        """Flush pending data for specified file"""
        if file_path not in self._pending_writes or not self._pending_writes[file_path]:
            return
        try:
            data = self._pending_writes.pop(file_path)
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            # Async write to file
            async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
                json_str = json.dumps(
                    data,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    check_circular=False,
                )
                await f.write(json_str)  # One JSON object per line
                self.total_written += 1

        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            # Keep data for retry next time

    async def _flush_all(self):
        """Flush pending data for all files"""
        for file_path in list(self._pending_writes.keys()):
            if self._pending_writes[file_path]:
                await self._flush_file(file_path)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            "is_running": self._is_running,
            "total_written": self.total_written,
            "queue_size": self.data_queue.qsize(),
            "files_pending": list(self._pending_writes.keys()),
        }
