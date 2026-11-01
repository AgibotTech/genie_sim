# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import threading
import time
import traceback


class TaskManager:
    def __init__(self, api_core, benchmark_config, benchmark_factory=None):
        self.api_core = api_core
        self.benchmark_config = benchmark_config
        self._benchmark_factory = benchmark_factory
        self._benchmark = None
        self._worker_thread = None
        self._worker_done = threading.Event()
        self._lock = threading.Lock()
        self._shutdown_reason = None
        self._stop_hook_started = False
        self._worker_status = "pending"
        self._worker_error = None
        self._started_at = None
        self._finished_at = None
        self._stop_hook = {"status": "not_requested"}

    def start(self):
        self.api_core.request_render()
        self.api_core._startup_render_held = True
        self._started_at = time.time()
        self._worker_status = "running"
        self._worker_thread = threading.Thread(
            target=self.worker,
            daemon=True,
            name="geniesim-task-worker",
        )
        self._worker_thread.start()

    def join(self, timeout=None):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        return self._worker_thread is None or not self._worker_thread.is_alive()

    def is_alive(self):
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def request_shutdown(self, reason):
        """Request the worker's own bounded stop path, once."""

        with self._lock:
            if self._shutdown_reason is None:
                self._shutdown_reason = str(reason)
            benchmark = self._benchmark
        if benchmark is not None:
            self._run_stop_hook(benchmark)
        return self.receipt()

    def receipt(self):
        with self._lock:
            worker = {
                "status": self._worker_status,
                "started_at_unix": self._started_at,
                "finished_at_unix": self._finished_at,
            }
            if self._worker_error is not None:
                worker["error"] = self._worker_error
            return {
                "reason": self._shutdown_reason,
                "stop_hook": dict(self._stop_hook),
                "worker": worker,
            }

    def _run_stop_hook(self, benchmark):
        with self._lock:
            if self._stop_hook_started or self._shutdown_reason is None:
                return
            self._stop_hook_started = True
            reason = self._shutdown_reason
            self._stop_hook = {"status": "running"}
        try:
            result = benchmark.request_shutdown(reason)
        except BaseException as error:
            self._record_worker_failure(error)
            with self._lock:
                self._stop_hook = {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
        else:
            with self._lock:
                self._stop_hook = {
                    "status": "requested",
                    "result": result,
                }

    def _record_worker_failure(self, error):
        if self.api_core.fatal_error is None:
            self.api_core.fatal_error = error
            self.api_core.fatal_exit_code = 1

    def worker(self):
        try:
            if self._benchmark_factory is None:
                from geniesim_benchmark.benchmark.task_benchmark import TaskBenchmark

                benchmark_factory = TaskBenchmark
            else:
                benchmark_factory = self._benchmark_factory
            benchmark = benchmark_factory(self.benchmark_config, self.api_core)
            with self._lock:
                self._benchmark = benchmark
                shutdown_pending = self._shutdown_reason is not None
            if shutdown_pending:
                self._run_stop_hook(benchmark)
            benchmark.evaluate_policy()
        except BaseException as error:
            self._record_worker_failure(error)
            with self._lock:
                self._worker_error = f"{type(error).__name__}: {error}"
                self._worker_status = "failed"
            traceback.print_exc()
        else:
            with self._lock:
                self._worker_status = "stopped" if self._shutdown_reason is not None else "completed"
        finally:
            with self._lock:
                self._finished_at = time.time()
            self._worker_done.set()
            self.api_core.stop()
