# Benchmark

See [Benchmark usage](benchmark_test_docs.md) for setup, task discovery and per-task smoke commands.

Isaac tasks run with system Python 3.12 and Isaac Sim 6.2 in the dedicated
`geniesim4:latest` image (`geniesim docker` / `geniesim docker6.2`). Newton and
libuipc tasks use the same interpreter; native MuJoCo keeps its separate runtime.

<a id="deformable-evaluation"></a>
<a id="deformable-scoring"></a>

Task launch and evaluation: [run a task](benchmark_test_docs.md#3-run-a-task).

Task-specific scoring: [scoop_popcorn rules](docs/scoop_popcorn_evaluation.md).

Current interfaces are defined by:

| Contract | Source |
|---|---|
| Task runtime selection | [runtime.py](src/geniesim_benchmark/runtime.py) |
| Isaac task parameters | [params.py](src/geniesim_benchmark/config/params.py) |
| EEF conversion and IK acceptance | [eef_control.py](src/geniesim_benchmark/utils/eef_control.py), [hard_ik.py](src/geniesim_benchmark/utils/hard_ik.py), [usd_ik.py](src/geniesim_benchmark/utils/usd_ik.py) |
| Scores and episode summaries | [eval_utils.py](src/geniesim_benchmark/plugins/output_system/eval_utils.py) |
| Inference preflight | [check-inference](skills/check-inference/SKILL.md) |

[License](LICENSE) · [Skills](skills/README.md) · [Challenge workflow](skills/robocoliseum/)
