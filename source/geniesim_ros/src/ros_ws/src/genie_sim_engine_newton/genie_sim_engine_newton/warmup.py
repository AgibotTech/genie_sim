# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Warp-kernel warmup for Newton.

Callers pre-compile Warp CUDA kernels for a given Newton solver before
the first real scene finalizes. This module intentionally has zero
top-level imports of Newton / Warp so importing it is cheap; the heavy
imports happen inside :func:`warmup_newton_kernels` on the first call.

Origins: this helper used to live at
``genie_sim_engine_isaacsim.kit.bootstrap._warmup_newton_kernels``
(kept there when Kit + Newton shared a package). After the Isaac /
Newton package split it moved here — the function itself is Kit-free,
and living in the Newton package removes the reverse dep the Newton
runtime script used to have on ``_isaacsim``.
"""

from __future__ import annotations


def warmup_newton_kernels(solver: str) -> None:
    """Pre-compile Warp CUDA kernels for Newton before the scene loads.

    Newton's ``ModelBuilder.finalize()`` triggers Warp JIT compilation of
    every solver's CUDA kernels.  This is a one-time cost (~5–30 s on first
    run; subsequent launches use the Warp kernel cache and take < 1 s).
    The problem: finalize is called mid-scene-load inside Isaac's GUI event
    loop, which blocks the UI thread and makes the viewport appear frozen.

    The fix: build a tiny 1-particle dummy model here — immediately after
    ``SimulationApp`` starts but before any stage opens — to force kernel
    compilation now.  The user sees a slow startup line in the log instead
    of a mid-run UI freeze.  The kernel cache survives across sessions so
    this overhead only occurs on first launch or after a Warp upgrade.

    The dummy model is discarded immediately; it has no effect on the scene.
    """
    print(
        f"[genie_sim_engine] warming up Newton/{solver} Warp kernels "
        f"(first-run: may take up to 30s; cached on next launch)…",
        flush=True,
    )
    try:
        import newton
        import warp as wp

        b = newton.ModelBuilder()
        # Register custom attributes for the requested solver so its
        # kernel-specific per-particle storage is included in the dummy model.
        solver_map = {
            "xpbd": "SolverXPBD",
            "style3d": "SolverStyle3D",
            "vbd": "SolverVBD",
            "mujoco-warp": "SolverMuJoCo",
            "featherstone": "SolverFeatherstone",
            "semiimplicit": "SolverSemiImplicit",
        }
        class_name = solver_map.get(solver.lower())
        cls = getattr(newton.solvers, class_name) if class_name else None
        if cls and hasattr(cls, "register_custom_attributes"):
            cls.register_custom_attributes(b)

        # Add a minimal cloth grid so finalize allocates particle buffers
        # and triggers the cloth/particle solver kernels.
        if solver.lower() in ("xpbd", "style3d", "vbd"):
            b.add_cloth_grid(
                pos=wp.vec3(0.0, 0.0, 0.0),
                rot=wp.quat(0.0, 0.0, 0.0, 1.0),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=2,
                dim_y=2,
                cell_x=0.1,
                cell_y=0.1,
                mass=0.01,
            )

        model = b.finalize("cuda:0")
        del model, b
        print("[genie_sim_engine] Newton kernel warmup complete.", flush=True)
    except Exception as exc:
        # Warmup failure is non-fatal — kernels will compile on first real
        # finalize instead (with the associated UI freeze on first run only).
        print(
            f"[genie_sim_engine] Newton kernel warmup skipped: {exc}",
            flush=True,
        )


__all__ = ["warmup_newton_kernels"]
