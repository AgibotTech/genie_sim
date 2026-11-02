# Benchmark

See [Benchmark usage](benchmark_test_docs.md) for setup, task discovery and per-task smoke commands.

<a id="deformable-evaluation"></a>
<a id="deformable-scoring"></a>

Task launch and evaluation: [run a task](benchmark_test_docs.md#3-run-a-task).

Current interfaces are defined by:

| Contract | Source |
|---|---|
| Task runtime selection | [runtime.py](src/geniesim_benchmark/runtime.py) |
| Isaac task parameters | [params.py](src/geniesim_benchmark/config/params.py) |
| Scores and episode summaries | [eval_utils.py](src/geniesim_benchmark/plugins/output_system/eval_utils.py) |
| Inference preflight | [check-inference](skills/check-inference/SKILL.md) |

[License](LICENSE) · [Skills](skills/README.md) · [Challenge workflow](skills/robocoliseum/)
