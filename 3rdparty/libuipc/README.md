# libuipc installation and optional source build

Use the prebuilt wheel by default. `geniesim bootstrap` and dedicated 6.2 Docker
builds select the SM89 or SM120 wheel from the asset repository's
`wheels/libuipc/`, verify its size and SHA256, and install it directly. They do
not compile libuipc. ModelScope is the default remote source, with automatic
HuggingFace fallback. See [wheel installation](../pyuipc-README.md) and
[download/cache configuration](../../docker/README.md#asset-backed-native-libuipc-installation).

```bash
# From the GenieSim code repository, using system CPython 3.12:
geniesim bootstrap --libuipc-only
# Or build the image for the deployment GPU:
geniesim docker6.2 build --build-arg LIBUIPC_SM=89
```

Before remote publication, set `GENIESIM_ASSETS_SRC` to the local asset checkout
containing `wheels/libuipc/`. A source build is optional and uses the procedure
below in a separate checkout.

## Source and patch order

Start from the [official libuipc repository](https://github.com/spiriMirror/libuipc)
at commit `292c98c3fec2dfbe9ae2c2407572f2f446a4a738`, then apply:

1. [0001-nvidia-optimizations.patch](patches/0001-nvidia-optimizations.patch) — NVIDIA runtime optimizations.
2. [0002-agibot-full-abd-cuda128-wheel.patch](patches/0002-agibot-full-abd-cuda128-wheel.patch) — Full ABD integration, CUDA 12.8 compatibility, and build/packaging tools.

The second patch depends on the first. They are plain Git diffs: use
`git apply`, rather than `git am`. Existing component copyright and
license notices are retained in the patches; applying them also supplies
`NOTICE`, `LICENSES/` and `tools/wheel_cuda128/THIRD_PARTY_NOTICES.md`.

Run this block from the GenieSim code repository root. The destination must not
already exist. The local commit records the complete patched source for the
packager; it does not need to be pushed.

```bash
set -e
GENIESIM_ROOT="$(git rev-parse --show-toplevel)"
PATCH_DIR="$GENIESIM_ROOT/3rdparty/libuipc/patches"
UIPC_BUILD_ROOT="$(dirname "$GENIESIM_ROOT")/libuipc-source-build"
mkdir -p "$UIPC_BUILD_ROOT"

git clone https://github.com/spiriMirror/libuipc "$UIPC_BUILD_ROOT/source"
cd "$UIPC_BUILD_ROOT/source"
git switch --detach 292c98c3fec2dfbe9ae2c2407572f2f446a4a738
git submodule update --init --recursive

git apply --whitespace=nowarn --check "$PATCH_DIR/0001-nvidia-optimizations.patch"
git apply --whitespace=nowarn "$PATCH_DIR/0001-nvidia-optimizations.patch"
git apply --whitespace=nowarn --check "$PATCH_DIR/0002-agibot-full-abd-cuda128-wheel.patch"
git apply --whitespace=nowarn "$PATCH_DIR/0002-agibot-full-abd-cuda128-wheel.patch"

git add -A
git commit -m "build(libuipc): apply NVIDIA and AgiBot CUDA 12.8 patches"
```

`--whitespace=nowarn` preserves the supplied source and license text. Keep the
baseline's pinned submodules; do not update them to their latest branches:
`external/muda` at `09f8a0beca898b5325c7b0c1e4cf67ea4781f3b9` and
`scripts/SymEigen` at `a12c56e34a95d501088e88c63034488e7c571c00`.

## Build inputs

Continue in that same shell and patched checkout. Provision these inputs before
running the build; the patch does not install system toolchains or drivers:

| Input | Requirement |
| --- | --- |
| Host and Python | Linux x86_64, CPython 3.12 with `venv` support |
| Native tools | GCC/G++ 11, CMake 3.26+, Ninja, binutils (`strip`), `patchelf` |
| `UIPC_CUDA_ROOT` | CUDA 12.8 toolkit root containing `bin/nvcc`, `bin/cuobjdump`, `include`, `lib64` and `version.json` |
| `UIPC_VCPKG_ROOT` | Provisioned vcpkg checkout containing `scripts/buildsystems/vcpkg.cmake` |
| `UIPC_DEPENDENCY_PREFIX` | Existing vcpkg `x64-linux` prefix with the dependencies from `tools/wheel_cuda128/vcpkg.json` and its pinned `vcpkg-configuration.json`, including headers, libraries and package copyright files |

The dependency prefix must include `libdwarf.a`, `libzstd.a`, `libz.a`,
`libconsole_bridge.a` and `libtinyxml.a`. `build_dependencies.py` consumes this
existing prefix and rebuilds the included cpptrace/urdfdom sources separately;
it does not provision the complete vcpkg prefix. The build copies the dependency
prefix instead of modifying it. Python build dependencies are installed into a
separate venv per architecture using the patch's pinned
`tools/wheel_cuda128/python-build-requirements.txt` and the configured pip index.

Set the three input variables in your shell to their actual directories, then:

```bash
: "${UIPC_CUDA_ROOT:?Set the CUDA 12.8 toolkit directory}"
: "${UIPC_VCPKG_ROOT:?Set the provisioned vcpkg checkout}"
: "${UIPC_DEPENDENCY_PREFIX:?Set the provisioned x64-linux dependency prefix}"
UIPC_WORK_ROOT="$UIPC_BUILD_ROOT/output"

python3.12 tools/wheel_cuda128/build_dependencies.py \
  --dependency-prefix "$UIPC_DEPENDENCY_PREFIX" \
  --work-dir "$UIPC_WORK_ROOT/third-party" \
  --cc gcc-11 --cxx g++-11 --jobs 8

python3.12 tools/wheel_cuda128/build_wheels.py \
  --arch both --work-dir "$UIPC_WORK_ROOT/native" \
  --cuda-root "$UIPC_CUDA_ROOT" \
  --dependency-prefix "$UIPC_DEPENDENCY_PREFIX" \
  --clean-dependency-lib "$UIPC_WORK_ROOT/third-party/install/lib" \
  --python python3.12 --vcpkg-root "$UIPC_VCPKG_ROOT"
```

`--arch both` builds separate SM89 and SM120 outputs. Use `--arch 89` or
`--arch 120` to build one target, and package only that target below. Each native
architecture output directory must be fresh. The recipe disables examples,
tests, benchmarks and GUI, and uses 32 native compile jobs. `--dry-run` prints
the native build plan without compiling; `--configure-only` prepares the venv
and CMake configuration without compiling.

## Package and install the locally built wheel

After the native build completes, package each target with its build venv:

```bash
for UIPC_ARCH in 89 120; do
  "$UIPC_WORK_ROOT/native/sm$UIPC_ARCH-cu128/venv/bin/python" \
    tools/wheel_cuda128/package_wheel.py \
    --arch "$UIPC_ARCH" \
    --build-dir "$UIPC_WORK_ROOT/native/sm$UIPC_ARCH-cu128/build" \
    --python "$UIPC_WORK_ROOT/native/sm$UIPC_ARCH-cu128/venv/bin/python" \
    --cuda-root "$UIPC_CUDA_ROOT" \
    --clean-dependency-lib "$UIPC_WORK_ROOT/third-party/install/lib" \
    --output-dir "$UIPC_WORK_ROOT/wheels" \
    --work-dir "$UIPC_WORK_ROOT/package-sm$UIPC_ARCH"
done

# Example: install the standard-named SM89 wheel into a separate Python 3.12 venv.
python3.12 -m venv "$UIPC_BUILD_ROOT/runtime-sm89"
"$UIPC_BUILD_ROOT/runtime-sm89/bin/python" -m pip install \
  "$UIPC_WORK_ROOT"/wheels/pyuipc-*sm89*.whl

# Direct pip installation does not create GenieSim's CUDA provider links.
# Use the same CUDA 12.8 toolkit for this process's runtime library lookup.
LD_LIBRARY_PATH="$UIPC_CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$UIPC_BUILD_ROOT/runtime-sm89/bin/python" - <<'PYUIPC'
import ctypes
from pathlib import Path
import uipc

ctypes.CDLL(str(Path(uipc.__file__).parent / "_native/libuipc_backend_cuda.so"))
print("libuipc import and CUDA backend load passed")
PYUIPC
```

Apply both patches to the pinned base in a fresh checkout; no additional C++
source edits are needed for the supplied CUDA 12.8 recipe. Set
`UIPC_CUDA_ROOT` to the actual CUDA 12.8 toolkit rather than a default `cuda`
symlink that may select a different release. The build script checks `nvcc`
12.8 and the requested GPU architecture; the packager also checks the toolkit
version and the compiler recorded in `CMakeCache.txt`.

The direct-pip example above uses a process-local library path. Apply that same
prefix when launching a custom libuipc application if CUDA 12.8 is not already
visible through the system dynamic loader. It checks import and backend
loading only; a matching GPU is still required to execute simulation.

Use fresh packaging work/output locations when repeating the procedure. The
packager emits standard `cp312-cp312-linux_x86_64.whl` names and `.sha256` files
under `$UIPC_WORK_ROOT/wheels`, and includes the patched source identity,
component notices and relevant source materials. Consult the supplied
`tools/wheel_cuda128/THIRD_PARTY_NOTICES.md` for component boundaries.

This source recipe produces CUDA 12.8 `licensefix2.pathfix1` versions. Rebuilding
does not guarantee byte-identical archives to the pinned asset release. GenieSim's automatic installer continues to use
the fixed versions and hashes in `docker/libuipc-artifacts.json`; a local build
does not replace that release automatically. Keep custom builds in their own
environment, or publish and explicitly pin a new artifact release for integration.

## Patch SHA256

| File | SHA256 |
| --- | --- |
| `0001-nvidia-optimizations.patch` | `d5d5b5632bc2ac04194927270ead778a95867129f1f1dd62cc8d704545ee5544` |
| `0002-agibot-full-abd-cuda128-wheel.patch` | `ef377382e88c618d3ebce78e29003e5436cf8f5a9043b95135bdf7bec5c25e13` |
