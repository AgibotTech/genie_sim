# Main IPC native runtime

The two supplied PDF-optimized native artifacts are stored with short names.
The current builds remove build-host paths; the short filenames preserve the
supplied wheel bytes, package versions and Apache license metadata.

| Repository file | GPU compute capability | CUDA runtime | Package version |
| --- | --- | --- | --- |
| `pyuipc-sm89.whl` | 8.9 / SM 89 | CUDA 13.0 | `0.9.0+pdfopt.ef5bfa2.sm89.cu130.licensefix1.pathfix1` |
| `pyuipc-sm120.whl` | 12.0 / SM 120 | CUDA 13.0 | `0.9.0+pdfopt.ef5bfa2.sm120.cu130.licensefix1.pathfix1` |

Install from the repository root:

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

Both Jazzy Dockerfiles preinstall the native wheel and its Python dependencies
with `install_libuipc.py --build-sm <SM>`. `geniesim docker build` (5.1 combo)
and `geniesim docker6.2 build` detect host GPU 0 and pass `LIBUIPC_SM` to Docker.
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
A complete CUDA 13 runtime must already be available. Both artifacts use the
system SDK's CUDA 13 libraries, including in the combo image with Isaac Sim 5.1.
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
Both backends link `libcudart.so.13`, `libcublas.so.13`,
`libcusolver.so.12` and `libcusparse.so.12`; a CUDA 12 runtime alone cannot
load it. Successful installation does not prove a native GPU step.

SHA256 (`pyuipc-sm89.whl`):

```
ff838f588f84c695a7fd41dd8769d8068f42d02987896e89882494e4a17e7c6d
```

SHA256 (`pyuipc-sm120.whl`):

```
3a7d0ea16b2221797d78b14de964042222569cd29f77e64e5d83654ff0e467f3
```

Both supplied replacement wheels carry the binary tag
`cp312-cp312-linux_x86_64`. Use only Linux x86_64 CPython 3.12; do not install
them into `omni_python` 3.11 or the Humble image's Python 3.10. Both manifests
record CUDA 13.0.2 builds targeting their respective GPU architectures.
The supplied build checks cover package integrity, decompressed GPU code,
build-host path removal, isolated installation and native loading for both
architectures. The SM89 build also passed a 10-frame native GPU simulation on
an RTX 4090 with driver 580.178.04 and PTX JIT disabled. SM120 native execution
remains unverified on matching hardware.

On 2026-10-27 the previous SM89 CUDA 13 `licensefix1` artifact passed cloth,
rigid and mixed GPU smoke cases on an RTX 4090 D with driver 580.159.03, for both a fresh container and
an upgrade from the CUDA 12 artifact. Each case advanced 30 steps and verified
finite state, gravity motion and `recover(0)`. These checks do not establish
policy task success, camera rendering or 60 Hz performance.
A rebuilt dependency layer also passed fresh startup, the same GPU cases and
restart reuse with Docker `--network none`, while pip indexes remained enabled.
Those historical checks do not qualify the current `pathfix1` artifacts.

Both artifacts require CUDA 13. The shared
evaluation environment must provide its matching CUDA runtime, cuBLAS,
cuSPARSE, cuSOLVER and nvJitLink libraries. The binary requires at least
GLIBC 2.35 and GLIBCXX 3.4.30 for SM89; Ubuntu 24.04 in the Jazzy image satisfies
these requirements. A host NVIDIA driver and `--gpus all` are still needed
to execute the solver. The installation check validates native loading after selecting a wheel from
the runtime GPU; it is not a substitute for a GPU step.

Both path-cleaned CUDA 13 replacements were supplied on 2026-10-30. Both contain an Apache
license classifier and `*.dist-info/licenses/LICENSE` matching the repository's
[preserved LICENSE](pyuipc-LICENSE). The previous GPLv3 classifier is absent.
The exact supplied bytes are identified by the SHA256 values above.

Each wheel includes `uipc/_native/build_manifest.json`, recording source commit
`ef5bfa21f343fdc0858c3040a6202634e466d747` from
https://github.com/kanghui0204/libuipc and per-library hashes. Both manifests
also record a CUDA 13 source override. These are supplied build records, not a
reproduction of the source build in this repository. Diagnostics use relative
or generic paths instead of build-host paths. The generic `/usr/src/libuipc`
root used for workspace path calculations does not need to exist at runtime.
The Python dependencies are unchanged. Upstream is https://github.com/spiriMirror/libuipc.

The Main-owned solver extension is separately derived from Apache-2.0 Newton code; see its
[solver provenance](../source/geniesim_ros/src/ros_ws/src/genie_sim_engine_libuipc/genie_sim_engine_libuipc/solver/PROVENANCE.md).
