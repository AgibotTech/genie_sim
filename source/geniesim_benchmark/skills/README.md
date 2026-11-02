# geniesim_benchmark — skills index

Each `SKILL.md` is a self-contained recipe an agent (or human) can follow end-to-end. `cat` the file to read it.

| Skill | When to use |
|---|---|
| [run-benchmark](run-benchmark/SKILL.md) | Launch a benchmark task locally against a user-provided inference server |
| [check-inference](check-inference/SKILL.md) | Probe a model inference WebSocket server and validate the response |
| [convert-dataset](convert-dataset/SKILL.md) | Convert robot-trajectory datasets between formats (agibot v1 → LeRobot v2.1) |
| [Challenge skills](robocoliseum/README.md) | Prepare models and data, submit jobs, run agents and diagnose connections |

For the canonical CLI / configs / scoring contract, see [`../../../.agent/geniesim_benchmark.md`](../../../.agent/geniesim_benchmark.md).
For build, launch and per-task tests, see [the usage guide](../benchmark_test_docs.md).
