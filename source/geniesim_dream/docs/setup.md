# Install Dream

Run every command below from `source/geniesim_dream/` in the enclosing
repository. Dream is installed separately from `geniesim bootstrap`. After
installation, follow [Models and assets](assets.md), then
[Running Dream](running.md).

## Requirements

| Environment | Python | Dependency source |
| --- | --- | --- |
| World model | 3.13 | Cosmos Framework's locked CUDA 12.8 inference group |
| Policy and bridge | 3.11 | OpenPI's lockfile, including PyTorch 2.7.1/CUDA 12.6 |
| VLM evaluator | 3.13 | PyTorch 2.10.0/CUDA 12.8 and the evaluation package |

Use a Linux machine with an NVIDIA GPU and compatible driver. The Framework
lockfile uses PyTorch and NVIDIA package indexes; OpenPI's lockfile fetches a
pinned LeRobot commit from GitHub. These hosts must be reachable or mirrored
during a fresh installation.

## Initialize dependencies

Initialize the pinned dependencies after cloning the enclosing repository:

```bash
git submodule update --init -- third_party
```

OpenPI's nested simulator submodules are not needed for these inference
services; leave them uninitialized.

## Create the environments

Install the environments from this directory so the three `.venv-*`
directories stay beside the release package. If either Python command is not
on `PATH`, substitute the full path to the matching interpreter.
If the interpreters are not installed and a standalone `uv` 0.11.3 or newer is
available, `uv python install 3.13 3.11` obtains them; create the environments
with
`uv venv --seed --python 3.13 .venv-world`,
`uv venv --seed --python 3.11 .venv-policy`, and
`uv venv --seed --python 3.13 .venv-evaluation`, then skip the three
`python -m venv` lines below.

The recipe pins `uv` to `>=0.11.3,<0.12` in the world environment:

```bash
export DREAM_PACKAGE_ROOT="$(pwd)"

# World model environment
python3.13 -m venv .venv-world
.venv-world/bin/python -m pip install 'uv>=0.11.3,<0.12'
(
  cd third_party/cosmos-framework
  UV_PROJECT_ENVIRONMENT="$DREAM_PACKAGE_ROOT/.venv-world" LD_LIBRARY_PATH= \
    "$DREAM_PACKAGE_ROOT/.venv-world/bin/uv" sync --frozen --no-default-groups --group cu128
)
.venv-world/bin/uv pip install --python .venv-world/bin/python -e .

# Policy environment
python3.11 -m venv .venv-policy
(
  cd third_party/openpi
  UV_PROJECT_ENVIRONMENT="$DREAM_PACKAGE_ROOT/.venv-policy" GIT_LFS_SKIP_SMUDGE=1 \
    "$DREAM_PACKAGE_ROOT/.venv-world/bin/uv" sync --frozen --no-default-groups
)
.venv-world/bin/uv pip install --python .venv-policy/bin/python -e services/policy

# Evaluation environment
python3.13 -m venv .venv-evaluation
.venv-world/bin/uv pip install --python .venv-evaluation/bin/python \
  'torch==2.10.0' 'torchvision==0.25.0' --torch-backend cu128
.venv-world/bin/uv pip install --python .venv-evaluation/bin/python -e services/evaluation
```

## Verify the installation

Verify the installed entry points and dependencies with:

```bash
for env in world policy evaluation; do
  .venv-world/bin/uv pip check --python ".venv-$env/bin/python"
done
.venv-world/bin/geniesim-dream --help
.venv-policy/bin/geniesim-dream-policy-server --help
.venv-evaluation/bin/geniesim-dream-evaluate --help
```

`lerobot` is a declared dependency of the pinned OpenPI checkout. If the
policy environment's `uv pip check` reports it missing, repeat the policy
`uv sync --frozen --no-default-groups` step above before running Dream.
