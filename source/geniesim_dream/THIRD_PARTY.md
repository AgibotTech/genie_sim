# Third-party source and attribution

`third_party/cosmos-framework` and `third_party/openpi` are unmodified Git
submodules. Their own licenses and notices apply; keep their histories when
updating them. The same URLs and commits are recorded in
[`third_party.json`](assets/manifests/third_party.json).

| Submodule | Pinned commit | License and notice files |
| --- | --- | --- |
| [NVIDIA Cosmos Framework](https://github.com/NVIDIA/cosmos-framework) | `ed8287fd7477113f8ac4f6b84290514d55cf0cdc` | `LICENSE` (OpenMDW-1.1), `NOTICE` |
| [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi) | `15a9616a00943ada6c20a0f158e3adb39df2ccac` | `LICENSE` (Apache-2.0), `LICENSE_GEMMA.txt` |

Dream's original platform and evaluation code, 20 example bundles and G2
kinematics URDF use [MPL 2.0](LICENSE). Policy adapter code extracted from an
internal closed-loop module retains its Apache-2.0 terms.

## Model origins

The released world model and task policy weights come from the
[ModelScope directory](https://www.modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset/tree/master/wm_checkpoints).
Their owners identify Cosmos3-Nano and OpenPI
`gs://openpi-assets/checkpoints/pi05_base` as starting points. The local
Cosmos3-Nano model card declares OpenMDW 1.1.

The included VAE comes from [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
revision `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`. The text tokenizer
and doll evaluator come from
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`; the default evaluator
is [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Their local model cards
declare Apache 2.0.
