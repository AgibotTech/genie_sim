# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import threading


class TaskManager:
    def __init__(self, api_core, benchmark_config):
        self.api_core = api_core
        self.benchmark_config = benchmark_config

    def start(self):
        worker_thread = threading.Thread(target=self.worker)
        worker_thread.start()

    def worker(self):
        from geniesim.benchmark.task_benchmark import main as benchmark_main

        benchmark_main(self.benchmark_config, self.api_core)
