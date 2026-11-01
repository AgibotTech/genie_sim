# geniesim_dream — Agent Development Guide

> Maintenance contract: update this guide and the README when service
> boundaries, public commands or action contracts change. Update the relevant
> files in `assets/manifests/` when model assets or third-party pins change.

## 1. What this is

`geniesim_dream` is a separately installed inference module under `source/`.
It is not part of the default `geniesim bootstrap` environment. The platform,
policy and evaluator each have their own Python environment under this package.
Follow the [installation guide](docs/setup.md) for Cosmos Framework's `cu128`
inference group and the OpenPI lockfile. Do not install Framework training
groups for this release.

## 2. Layout

- `src/geniesim_dream/`: sessions, HTTP API, UI and world model runtime.
- `services/policy/`: OpenPI policy serving, bridge and action conversion.
- `services/evaluation/`: VLM service and task scoring flows.
- `examples/`: all 20 reviewed bundles.
- `assets/stats/`: public G1/G2 numeric action stats without dataset paths.
- `assets/robot/urdf/`: committed G2 kinematics URDF for policy forward kinematics (no meshes).
- `third_party/`: pinned Git submodules with their original history.
- `src/geniesim_dream/cosmos_adapter.py`: G1/G2 batch and action-position adapter
  for the unmodified NVIDIA Framework commit `ed8287f`.

## 3. CLI and API surface

| Command | Purpose |
| --- | --- |
| `geniesim-dream check --config ...` | Validate assets and discover examples |
| `geniesim-dream serve --config ...` | Serve world model sessions and UI |
| `geniesim-dream-policy-server` | Serve a pi0.5 checkpoint through OpenPI |
| `geniesim-dream-policy-bridge` | Convert policy outputs to Dream actions |
| `geniesim-dream-evaluate` | Run visual task evaluation |

## 4. Architectural rules

1. Keep inference and scoring independent of training entry points, datasets
   and experiment configs.
2. Preserve the 29D action order: `head_rel(9), right_rel(9), right_grip(1),
   left_rel(9), left_grip(1)`. Keep the fixed 4-frame history, checkpoint stats SHA
   checks and domain IDs: `agibotworld_g1` (15), `agibotworld_g2` (17). The
   example loader maps the older `agibotworld` name to G1. Each chunk defaults
   to 32 action rows; the API also accepts 16. Policy `norm_stats.json` and
   world-model `stateend` stats are distinct. Check numerical results when
   changing conversions.
3. Share one world model instance while keeping history separate per session.
   Verify a closed loop with the real world-model, policy and evaluator services.
4. Run policy and evaluator in separate environments. Dream calls `/predict`
   and `/evaluate`; the policy bridge connects to OpenPI over WebSocket.
5. Keep Cosmos Framework pinned to the unmodified public `ed8287f` commit and
   OpenPI as an upstream Git submodule. Retain their license and notice files.
   Keep Dream batch logic in the adapter; when updating Framework, recheck its
   `_pack_input_sequence` result and vision spans.

## 5. Skills index

| Skill | When to use |
| --- | --- |
| [run-dream](skills/run-dream/SKILL.md) | Run a G1/G2 Dream rollout with prepared environments and checkpoints |

See [skills/README.md](skills/README.md) for the package skill index.

## 6. Common workflows

Use [run-dream](skills/run-dream/SKILL.md) for the operator workflow. Before a
Framework update, compare the adapter's batch and packed action positions.
For OpenPI updates, verify the upstream revision and update the submodule
without merging local code into its checkout.

## 7. Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing stats or SHA mismatch | `assets/stats/` and `world_model_stats_pin.json` |
| Policy does not load | OpenPI commit, checkpoint assets and action space marker |
| G2 FK fails | `GENIESIM_DREAM_G2_URDF` points at the kinematics URDF |
| VLM is unavailable | Model config paths, evaluator health and shared output root |

## 8. Verification recipes

After placing the model assets, validate them and the example bundles from
`source/geniesim_dream/`:

```bash
export DREAM_PACKAGE_ROOT="$PWD"
export DREAM_ASSET_ROOT="${DREAM_ASSET_ROOT:-$PWD/checkpoints}"
export DREAM_OUTPUT_DIR="${DREAM_OUTPUT_DIR:-$PWD/outputs}"
.venv-world/bin/geniesim-dream check --config configs/dream.example.json
```

For an end-to-end check, use [run-dream](skills/run-dream/SKILL.md) and record
the session ID, settings, `vision.mp4`, `actions.npy` and VLM report.

## 9. Do not

- Do not copy private training scripts, experiment paths or dataset metadata.
- Record runtime validation with the real services and output artifacts.
- Do not replace a third-party Git history with copied files.
