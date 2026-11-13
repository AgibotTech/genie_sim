# Main IPC native runtime

**Default: install the prebuilt wheel directly** through `geniesim bootstrap`
or a dedicated 6.2 Docker build. Building libuipc is optional. For the official source
commit, the two ordered patches and build/package commands, see
[the source build guide](libuipc/README.md).

The universal PDF-optimized native artifact lives in the asset repository at
`wheels/libuipc/pyuipc.whl`. One archive contains SM89 and SM120 native CUDA
code; installation and Docker builds do not query the host GPU.
The code repository retains its license and provenance and pins its size and
SHA256 in `docker/libuipc-artifacts.json`.

| Asset release file | Native GPU targets | CUDA runtime | Package version |
| --- | --- | --- | --- |
| `pyuipc.whl` | SM89 and SM120 | CUDA 12.8 | `0.9.0+pdfopt.ef5bfa2.sm89sm120.cu128.licensefix2.pathfix1` |

The short asset filename leaves the supplied wheel bytes and embedded metadata
unchanged. New universal source builds use a standard wheel filename without
`smXX` in the version; see the source build guide.

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

The dedicated 6.2 Dockerfile preinstalls the universal CUDA 12.8 wheel and its
Python dependencies with `install_libuipc.py --build-image`. The default
`geniesim docker build` and `geniesim docker6.2 build` require no GPU or
architecture build argument. The `geniesim docker5.1` combo omits native
libuipc during image construction and automatic startup installation.

```bash
geniesim docker6.2 build
```

The image records the artifact hash. At startup, the installer verifies that
artifact, its dependencies and native loading without pip or GPU detection.
A broken or outdated marked installation requires rebuilding the image.
Unmarked environments retain the normal bootstrap install/repair path.
`--dependencies-only` prepares Python dependencies; `--download-only` caches
and verifies the archive without runtime ABI or native loading checks.

The installer restores the standard wheel filename in a temporary copy for
installation. Do not pass the short asset filename directly to pip.
`--no-deps` skips Python dependencies in an already prepared environment.
Newton is not a dependency of this native wheel. The archive contains
`pyuipc.cpython-312-x86_64-linux-gnu.so` and requires Linux x86_64 CPython 3.12.
Simulation requires an SM89 or SM120 GPU and a compatible NVIDIA driver.
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
The backend links `libcudart.so.12`, `libcublas.so.12`,
`libcusolver.so.11` and `libcusparse.so.12`. Successful installation does not prove a native GPU step.

SHA256 (`pyuipc.whl`):

```
1c07165a5f92292bb0703d66af44fddd44bf738fee479c77673253c9e52669fb
```

The wheel carries the binary tag `cp312-cp312-linux_x86_64`; do not install it
into `omni_python` 3.11 or Humble's Python 3.10. Its manifest records CUDA
Toolkit 12.8, nvcc 12.8.61 and both native GPU targets. A compatible driver and
`--gpus all` are needed to execute the solver. Installation and dynamic-load
checks do not establish GPU simulation acceptance.

Each wheel includes `uipc/_native/build_manifest.json` with source, toolchain
and per-library hashes. The official base is
`292c98c3fec2dfbe9ae2c2407572f2f446a4a738` from
https://github.com/spiriMirror/libuipc. The patched source and recipe identity
is `8a75cae5a92a7b859a6296bd1a7be3f5f83f8f43`; this local commit identity does
not imply that it is published in the upstream repository. The build recipe
recorded in the manifest is `tools/wheel_cuda128/package_wheel.py`.

The retained [upstream libuipc license](pyuipc-LICENSE) is unchanged and covers
the original upstream Apache-2.0 terms. It does not describe every component
inside the supplied wheel.
The archive retains `NOTICE`, original upstream license texts, third-party
notices and covered source under `*.dist-info/licenses/`. Preserve those
materials when using or redistributing the artifacts. Package classifiers
alone do not describe all component licenses.

The Main-owned solver extension is separately derived from Apache-2.0 Newton code; see its
[solver provenance](../source/geniesim_ros/src/ros_ws/src/genie_sim_engine_libuipc/genie_sim_engine_libuipc/solver/PROVENANCE.md).
