# ROS workspace roadmap

This document records unimplemented work. Entries are scope and acceptance gates,
not release commitments. Current APIs and support are documented in the package
references and [runtime acceptance](ISAACSIM_ACCEPTANCE.md).

| Area | Work and acceptance gate |
| --- | --- |
| Joint overlay URDF application | General patch library and CLI for link inertials and URDF-expressible fields; preserve an unchanged empty-stack output, validate physical inertia, and include selected overlays in cache identity |
| Joint parameter data | Evaluate consolidation of xacro selectors and sidecars only after the complete inertial/application path is verified |
| Newton collision representation | Evaluate an end-to-end SDF representation/import/adapter path with room-scale performance and compiled-geometry validation; this does not replace spawn/contact checks |
| Command delay | Define and test a consumer separately from the existing joint-property resolver; schema storage alone is insufficient |
| Client committed stepping | Paused startup handshake, command identity shared with the latch, K-tick acknowledgement, reset/simulation-time identity, timeout status and render frame correlation; requires core contract review |
| Benchmark client integration | Application-owned grouped actions/observations, policy lifecycle, checkers, scoring, reports and CLI dispatch; validate cross-process DDS, real Newton, rendering and task results |

These entries do not change current backend support or the acceptance matrix.
