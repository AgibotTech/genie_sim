# Third-party materials

Agent2Act's original source code is licensed under the Mozilla Public License
Version 2.0; see [LICENSE](LICENSE). This follows the license used by Genie Sim's
`source/geniesim` module. The root license text is an unchanged copy of that module's
license, checked at Genie Sim commit `f62fb17b774824684bb20417cc869bcc1e61ec73`.

This inventory records separate terms and evidence for bundled third-party
materials. The project license does not replace their applicable terms.
USD-derived geometry and calibration data also require confirmation of their
source's applicable terms.

## Genie Sim descriptions

Source: [AgibotTech/genie_sim](https://github.com/AgibotTech/genie_sim), commit
`6ca11c7593ecf6b7dae58c28fd19f4a789a0dd46` (verified 2026-10-26).
Upstream attribution: AgiBot Inc., Genie Sim Team.

The following files are byte-identical copies from that commit:

| Bundled file | Upstream path | License evidence |
|---|---|---|
| `examples/urdf/G2_NO_GRIPPER.urdf` | `source/geniesim_benchmark/src/geniesim_benchmark/utils/IK-SDK/G2_NO_GRIPPER.urdf` | `source/geniesim_benchmark/LICENSE`: MPL-2.0 |
| `examples/urdf/G2_omnipicker_fixed_dual.urdf` | `source/data_collection/config/robot_cfg/G2/G2_omnipicker_fixed_dual.urdf` | Root `LICENSE` explicitly covers `source/data_collection`; module `LICENSE.txt`: MPL-2.0 |

The Mozilla Public License Version 2.0 is reproduced in [LICENSES/MPL-2.0.txt](LICENSES/MPL-2.0.txt).
Retain that license and applicable notices with these files and make covered source
and modifications available as required by MPL-2.0.
