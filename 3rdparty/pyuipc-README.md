# Main IPC native runtime

**Default: install the prebuilt wheel directly** through `geniesim bootstrap`
or a dedicated 6.2 Docker build. Building libuipc is optional. For the official source
commit, the two ordered patches and build/package commands, see
[the source build guide](libuipc/README.md).

The two supplied PDF-optimized native artifacts now live in the asset repository
under `wheels/libuipc/` with short names.
The code repository retains their license and provenance, plus the pinned
`docker/libuipc-artifacts.json`; it no longer contains the wheel bytes.
The current builds remove build-host paths; the short filenames preserve the
wheel bytes, package versions and embedded component license materials.

| Asset release file | GPU compute capability | CUDA runtime | Package version |
| --- | --- | --- | --- |
| `pyuipc-sm89.whl` | 8.9 / SM 89 | CUDA 12.8 | `0.9.0+pdfopt.ef5bfa2.sm89.cu128.licensefix2.pathfix1` |
| `pyuipc-sm120.whl` | 12.0 / SM 120 | CUDA 12.8 | `0.9.0+pdfopt.ef5bfa2.sm120.cu128.licensefix2.pathfix1` |

Install from the code repository root. Before remote publication, set
`GENIESIM_ASSETS_SRC` to the asset checkout containing `wheels/libuipc/`.
The installer verifies and caches local bytes, or downloads the matching
HF/ModelScope publication. See
[asset-backed installation](../docker/README.md#asset-backed-native-libuipc-installation).

```bash
geniesim bootstrap
# Retry only the native installation, without peer installation or prompts:
geniesim bootstrap --libuipc-only
```

`geniesim bootstrap` invokes this installer by default with `--if-needed` in
the same Python environment. It reuses an installed artifact only when its
version and archive hash match, then checks native library loading. Failure
warns without blocking required module installation. Manual `--if-needed`
uses the same check. Installed dependency versions are constrained so the
native installation cannot silently upgrade or downgrade the shared SDK stack.
When apt and pip expose multiple versions, only the version selected by the
current interpreter is constrained. A non-root user can reuse a healthy
root-installed artifact without needing write access to its installation lock.

The dedicated 6.2 Dockerfile preinstalls the selected CUDA 12.8 wheel and its Python
dependencies with `install_libuipc.py --build-sm <SM>`. The default
`geniesim docker build` and `geniesim docker6.2 build` select dedicated 6.2;
these commands detect host GPU 0 and pass `LIBUIPC_SM` to Docker.
The `geniesim docker5.1` combo omits native libuipc during image construction
and automatic startup installation.
For a build host without a supported GPU, select the deployment architecture:

```bash
geniesim docker6.2 build --build-arg LIBUIPC_SM=89
# For SM120 deployment: --build-arg LIBUIPC_SM=120
```

Docker itself needs no GPU during construction. Dependencies include `numpy`,
`matplotlib`, `polyscope>=2.5,<3` and `huggingface_hub` for `uipc.assets`.
The image records its architecture and artifact hash. At startup, the installer
still detects the GPU, then verifies the preinstalled artifact, dependencies and
native loading without pip. A different GPU architecture or broken installation
requires rebuilding the image. Pip index access remains enabled. IPC with local
assets needs no package downloads; remote asset retrieval still needs network.
Environments without an image marker retain the normal bootstrap install/repair
path. `--dependencies-only` remains available for preparing Python dependencies.

The script defaults to `nvidia-smi` GPU 0; `--gpu N` selects another index.
It rejects every SM except 89 and 120 before pip, verifies SHA256, then restores
the standard wheel filename in a temporary copy for installation. Do not pass
the short filenames directly to pip. `--no-deps` skips the wheel's Python
dependencies if they are already installed. Newton is not a dependency of these
native wheels. One Python environment holds one selected architecture's build.

Both artifacts contain `pyuipc.cpython-312-x86_64-linux-gnu.so` and require
Linux x86_64 CPython 3.12, a compatible NVIDIA driver and a matching GPU.
A complete CUDA 12.8 runtime must already be available. The dedicated Isaac
Sim 6.2 image provides it through the bundled SDK. The installer also supports
a complete matching system-Python or system-toolkit provider.
The installer links a complete provider into `uipc/_native`, where the backend's
`$ORIGIN` RUNPATH finds it; no manual `LD_LIBRARY_PATH` is needed. These links
are registered in RECORD, leaving native library bytes and SDK packages intact.
CUDA 12 and CUDA 13 providers are kept separate because cuSPARSE shares its
SONAME across those releases. Existing artifacts that fail native loading are
reinstalled through the same staged validation and rollback path.
Upgrades copy the complete old package and metadata before removing them,
preserving symbolic links without renaming image-layer directories. Failed
publication or native validation restores the old installation. If rollback
itself fails, the installer reports and retains its recovery directory.
The installer also accepts a working system dynamic loader configuration;
it does not download CUDA runtime libraries.
Both backends link `libcudart.so.12`, `libcublas.so.12`,
`libcusolver.so.11` and `libcusparse.so.12`. Successful installation does not prove a native GPU step.

SHA256 (`pyuipc-sm89.whl`):

```
13d0c5b79352b54ae9d06a8e074564fb9860b74d13072928a4f9592606721323
```

SHA256 (`pyuipc-sm120.whl`):

```
c332124443290731dbe7ad31436e40ca8027424758aa9be98be735dbce22133f
```

Both wheels carry the binary tag `cp312-cp312-linux_x86_64`. Use only Linux
x86_64 CPython 3.12; do not install them into `omni_python` 3.11 or the Humble
image's Python 3.10. Their manifests record CUDA Toolkit 12.8 and nvcc 12.8.61,
with separate SM89 and SM120 targets. A compatible host NVIDIA driver and
`--gpus all` are needed to execute the solver. An installation or dynamic-load
check does not establish native GPU simulation or matching-hardware acceptance.

Each wheel includes `uipc/_native/build_manifest.json` with source, toolchain
and per-library hashes. The official base is
`292c98c3fec2dfbe9ae2c2407572f2f446a4a738` from
https://github.com/spiriMirror/libuipc. The patched source and recipe identity
is `f2cc69eeaa87e8a0919857298b06967050dd7fb8`; this local commit identity does
not imply that it is published in the upstream repository. The build recipe
recorded in the manifest is `tools/wheel_cuda128/package_wheel.py`.

The retained [upstream libuipc license](pyuipc-LICENSE) is unchanged and covers
the original upstream Apache-2.0 terms. It does not describe every component
inside the supplied wheels.
The archives retain `NOTICE`, original upstream license texts, third-party
notices and covered source under `*.dist-info/licenses/`. Preserve those
materials when using or redistributing the artifacts. Package classifiers
alone do not describe all component licenses.

The Main-owned solver extension is separately derived from Apache-2.0 Newton code; see its
[solver provenance](../source/geniesim_ros/src/ros_ws/src/genie_sim_engine_libuipc/genie_sim_engine_libuipc/solver/PROVENANCE.md).
