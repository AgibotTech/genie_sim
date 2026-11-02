# SolverIPC attribution and ownership

This directory vendors the Newton IPC solver extension, not Newton itself.
The original 2026 Newton copyright and Apache-2.0 SPDX headers remain in both
source files. [LICENSE](LICENSE) preserves the complete upstream license.
The supplied `solver_ipc.py` content is identified by SHA-256
`0a85a17029901cca70592a6f36fc9ec362a83479dc1b7d113609a7d23aad7972`.

The installed extension is
`genie_sim_engine_libuipc.solver.SolverIPC`. It imports public Newton types and
`newton.solvers.SolverBase`; it does not register or monkey-patch a solver in
`newton.solvers`. Newton, Warp, MuJoCo and MuJoCo-Warp remain installed SDK
distributions rather than vendored copies.

The local extension preserves input poses/velocities for bodies omitted from
native readback, so visual kinematic arms retain their FK pose. Native solved
rows override that fallback. Construction/reset use the native initial state.
Native installation and the model/state boundary are documented in the
[backend design](../../DESIGN.md).
