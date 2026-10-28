# Scene YAML `newton` keys

These keys live in the **scene YAML** `newton` block. The launcher may
provide scene defaults; explicit scene values win. There are no environment
or flag-file overrides for these policies.

| Key | Default | Purpose |
|---|---|---|
| `newton.mjc_contact.robot.friction` | RL task generator: `[2.0, 0.5, 0.2]` | Robot contact friction the RL task generator writes. Explicit task contact values override these defaults. These are MuJoCo sliding/torsional/rolling coefficients, not PhysX static/dynamic/restitution. |
| `newton.eval_static_props` | `[]` | Checker path allowlist copied from the task scene's `eval_static_props`; used to compute world min/max boxes on the Newton composed stage. |

The RL profile already has `usd_drive_api.gripper.master_stiffness=100` and
the table collider exclusion policy. None of these keys change the generic
profile, and the MuJoCo contact numbers are not translated into PhysX
parameters.

Static world boxes publish through `setup/static_facts.py` and the retained
`benchmark_publisher.py` writer on `/rlinf/static_prop_aabb` (String JSON:
`names` and `aabb_world`, aligned rows of six min/max values). Both ends use
reliable, transient-local QoS. Empty publishes clear the scene's facts;
shutdown destroys the retained writer. The body-local center/half-extent
topic stays separate. Newton's replicated worlds share its background frame;
fused keeps its existing per-environment world boxes.
