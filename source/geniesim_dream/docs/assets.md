# Model and runtime assets

Download model assets into `DREAM_ASSET_ROOT` (by default, this package's
Git-ignored `checkpoints/` directory). Set it explicitly for another location.
The `fd_cosmos/` and `policy/` model directories are available in the
[ModelScope `wm_checkpoints` directory](https://www.modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset/tree/master/wm_checkpoints).

## What you need

The one-click launcher starts all four services, including the G1 bread policy
and Qwen3.5 evaluator. Prepare these files even if your first session replays
recorded actions:

| Asset under `DREAM_ASSET_ROOT` | Source | When needed |
| --- | --- | --- |
| `fd_cosmos/` | ModelScope directory above | Every world-model run; includes the Wan VAE and Qwen text tokenizer |
| `policy/g1-bread-v1/` | ModelScope `policy/` directory | Required when the one-click launcher starts, even for action replay |
| `vlm/qwen3.5/` | [Qwen3.5 download command](#evaluator-models) | One-click launcher startup and bread, towel or cloth evaluation |
| Other `policy/<task>/` directories | ModelScope `policy/` directory | Loaded only when you select that task's policy |
| `vlm/qwen3-vl/` | [Qwen3-VL download command](#evaluator-models) | Doll evaluation |

OpenPI fetches its PaliGemma tokenizer on first policy use. For offline runs,
prepare its [local cache](#policy-tokenizer-for-offline-runs) beforehand.

## Download and place files

From `source/geniesim_dream/`, put the complete ModelScope `fd_cosmos/` and at
least `policy/g1-bread-v1/` under `DREAM_ASSET_ROOT`. Add other policy
directories for their tasks. The G1/G2 stats and
G2 robot file already ship with this source package. `fd_cosmos/` already
contains the Wan VAE and five Qwen3-VL text tokenizer files. `checkpoints/`
may itself be a symlink to another asset directory. Set the paths before
downloading evaluator models or checking files:

```bash
export DREAM_PACKAGE_ROOT="$(pwd)"
export DREAM_ASSET_ROOT="${DREAM_ASSET_ROOT:-$DREAM_PACKAGE_ROOT/checkpoints}"
mkdir -p "$DREAM_ASSET_ROOT"
```

### Evaluator models

Install the official `hf` CLI in Dream's world environment:

```bash
.venv-world/bin/uv pip install --python .venv-world/bin/python 'huggingface_hub==0.36.2'
```

The launcher uses Qwen3.5 by default for bread, towel, cloth and generic
evaluation:

```bash
.venv-world/bin/hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir "$DREAM_ASSET_ROOT/vlm/qwen3.5"
```

Download the full Qwen3-VL model only for doll evaluation:

```bash
.venv-world/bin/hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --local-dir "$DREAM_ASSET_ROOT/vlm/qwen3-vl"
```

### Policy tokenizer for offline runs

OpenPI fetches `gs://big_vision/paligemma_tokenizer.model` on first policy use.
For offline runs, place it as a regular file at
`$DREAM_ASSET_ROOT/policy/openpi_cache/big_vision/paligemma_tokenizer.model`.
The pinned OpenPI version does not support a symlink outside this cache.

## File layout

The ModelScope trees total about 72 GB: `fd_cosmos/` is about 34.3 GB, and
each of the five policy checkpoints is about 7.5 GB.

Supported asset layout (only `fd_cosmos/`, `policy/g1-bread-v1/` and
`vlm/qwen3.5/` are required for the first run):

```text
geniesim_dream/checkpoints/  (= $DREAM_ASSET_ROOT)
  fd_cosmos/
    config.json, safetensors index and shards, world_model_stats_pin.json
    vae/Wan2.2_VAE.pth
    tokenizer/Qwen3-VL-8B-Instruct/  five text tokenizer files
  policy/
    g1-bread-v1/              model.safetensors, assets/dream/norm_stats.json
    g1-towel-v1/              model.safetensors, assets/dream/{norm_stats,action_space}.json
    g1-doll-in-v1/            model.safetensors, assets/dream/norm_stats.json
    g1-doll-out-v1/           model.safetensors, assets/dream/norm_stats.json
    g2-cloth-v1/              model.safetensors, assets/dream/{norm_stats,action_space}.json
  policy/openpi_cache/big_vision/paligemma_tokenizer.model  optional local OpenPI cache
  vlm/qwen3-vl/              VLM files for doll evaluation
  vlm/qwen3.5/               Default VLM; bread, towel, cloth and generic evaluation
geniesim_dream/assets/robot/urdf/
  G2_t2v2_crsB_ctek90d.urdf   kinematics-only; no mesh binaries shipped
```

Continue with [Running Dream](running.md) after placing the required files.
The check below is optional and can run before launch.

## Optional integrity check

The runtime checks `world_model_stats_pin.json` against the committed G1/G2
stats. To verify the downloaded world-model files, run:

```bash
cd "$DREAM_ASSET_ROOT/fd_cosmos"
sha256sum -c "$DREAM_PACKAGE_ROOT/assets/manifests/g1-g2-mixed-v1.sha256"
```

This manifest covers the seven weight shards, index, stats pin, VAE and five
text tokenizer files. Other [SHA256 manifests](../assets/manifests/) cover the
policy, evaluator and G2 URDF assets. Policy `metadata.pt`, `config.json` and
download caches are outside these checks. Model origins and
terms are recorded in [third-party attribution](../THIRD_PARTY.md#model-origins).
